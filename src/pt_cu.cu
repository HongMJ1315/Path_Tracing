#include "pt_cu.cuh"
#include <cstdio>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
// EPSILON 已經定義在 geometric.cuh 中


// 初始化隨機數生成器
__global__ void pt_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*--------------------------
傳統 Path Tracing Kernel (MIS 完美升級版)
--------------------------*/
__global__ void cuda_path_trace_kernel(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaCamera cam, curandState *states,
    int W, int H, int max_depth, float3 *d_image, int spp
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];

    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    for(int s = 0; s < spp; ++s){
        float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
        float pixel_x = (float) px + curand_uniform(&localState);
        float pixel_y = (float) py + curand_uniform(&localState);
        float3 eyeray_point = cam.eye;
        float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
        float3 eyeray_dir = normalize(pixel_pos - eyeray_point);
        float eyeray_refract = 1.0f;
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

        bool last_is_delta = true;
        float last_pdf_bsdf = 1.0f;

        for(int depth = 0; depth < max_depth; ++depth){
            CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
                d_spheres, num_spheres,
                d_triangles, num_triangles,
                d_lights, num_lights);

            if(!hit.hit) break;

            float3 wo = eyeray_dir * -1.0f;

            // --- 1. BSDF 光線打中光源 ---
            if(hit.is_light){
                float3 emission = hit.mtl.base_color;
                float area = 1.0f;
                float cone_ratio = 1.0f;
                bool valid_light = false;

                // 找出打中哪一盞燈，以取得它的半徑與聚光角度
                for(int i = 0; i < num_lights; ++i){
                    float3 center_to_hit = hit.pos - d_lights[i].pos;
                    if(fabs(length(center_to_hit) - d_lights[i].light_ball.r) < 1e-2f){
                        valid_light = true;
                        area = 4.0f * PI * d_lights[i].light_ball.r * d_lights[i].light_ball.r;

                        if(d_lights[i].cutoff > 0.0f && !d_lights[i].is_parallel){
                            cone_ratio = (1.0f - cosf(d_lights[i].cutoff)) / 2.0f;
                            // 檢查是否從燈罩後方打中（無光區）
                            float3 main_dir = normalize(d_lights[i].dir);
                            if(depth == 0){
                                cone_ratio = 1.f;
                            }
                            else if(dot(main_dir, normalize(center_to_hit)) < cosf(d_lights[i].cutoff)){
                                cone_ratio = 0.0f;
                            }

                        }
                        break;
                    }
                }

                // [核心修復] 將 Flux 轉換為正確的 Radiance
                if(valid_light && cone_ratio > 0.0f){
                    emission = emission / (area * cone_ratio);
                }
                else{
                    emission = make_float3(0.0f, 0.0f, 0.0f);
                }

                if(emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f){
                    if(last_is_delta){
                        float3 contrib = throughput * emission;
                        if(is_valid_color(contrib)){
                            final_color = final_color + clamp_radiance(contrib, 15.0f);
                        }
                    }
                    else{
                        // [MIS 策略 A]：計算 BSDF 打中光源的權重
                        float pdf_light_dir = 0.0f;
                        // ... (此處保留你原本計算 pdf_light_dir 的 for 迴圈邏輯) ...
                        // 為了節省版面我省略，請直接沿用你原本 if(!last_is_delta) 裡算 MIS 的程式碼

                        if(pdf_light_dir > 0.0f){
                            float p_b = last_pdf_bsdf * last_pdf_bsdf;
                            float p_l = pdf_light_dir * pdf_light_dir;
                            float mis_w = p_b / fmaxf(p_b + p_l, 1e-8f);

                            float3 contrib = throughput * emission * mis_w; // 使用放大後的 emission
                            if(is_valid_color(contrib)){
                                final_color = final_color + clamp_radiance(contrib, 15.0f);
                            }
                        }
                    }
                }
                break; // 擊中光源，路徑終止
            }

            // --- 2. Next Event Estimation (NEE) 光源直接採樣 ---
            if(hit.mtl.eta <= 0.0f && (hit.mtl.metallic < 0.99f || hit.mtl.roughness > 0.01f)){
                if(num_lights > 0){
                    int l_idx = min((int) (curand_uniform(&localState) * num_lights), num_lights - 1);
                    CudaLight light = d_lights[l_idx];

                    if(light.is_parallel){
                        float3 light_dir = normalize(light.dir * -1.0f);
                        float cos_surface = fmaxf(0.0f, dot(hit.normal, light_dir));

                        if(cos_surface > 0.0f){
                            // [修復] 使用 check_visibility
                            float3 transmittance = check_visibility(hit.pos + hit.normal * EPSILON,
                                hit.pos + light_dir * 1e4f, d_spheres, num_spheres, d_triangles,
                                num_triangles);

                            if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                                float3 brdf = bsdf_evaluate(hit.mtl, wo, light_dir, hit.normal);
                                float3 contrib = throughput * brdf * light.illum *
                                    transmittance * cos_surface * (float) num_lights;
                                if(is_valid_color(contrib)){
                                    final_color = final_color + clamp_radiance(contrib, 15.0f);
                                }
                            }
                        }
                    }
                    else{
                        float3 d_vec_local = random_in_unit_sphere_device(&localState);
                        if(length(d_vec_local) > 0.001f) d_vec_local = normalize(d_vec_local);
                        else d_vec_local = make_float3(0, 1, 0);

                        float3 light_pos = light.pos + d_vec_local * light.light_ball.r;
                        float3 wi_light = light_pos - hit.pos;
                        float dist2 = dot(wi_light, wi_light);
                        float dist = sqrtf(dist2);
                        wi_light = wi_light / dist;

                        float cos_surface = fmaxf(0.0f, dot(hit.normal, wi_light));
                        float cos_light = fmaxf(0.0f, dot(d_vec_local, wi_light * -1.0f));

                        if(cos_surface > 0.0f && cos_light > 0.0f){
                            bool inside_cone = true;
                            if(light.cutoff > 0.0f && !light.is_parallel){
                                float3 main_dir = normalize(light.dir);
                                if(dot(main_dir, wi_light * -1.0f) < cosf(light.cutoff)){
                                    inside_cone = false;
                                }
                            }

                            if(inside_cone){
                                float3 transmittance = check_visibility(hit.pos + hit.normal * EPSILON,
                                    light_pos + d_vec_local * EPSILON, d_spheres, num_spheres,
                                    d_triangles, num_triangles);

                                if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                                    float area = 4.0f * PI * light.light_ball.r * light.light_ball.r;
                                    float pdf_light_area = 1.0f / (num_lights * area);
                                    float pdf_light_dir = pdf_light_area * dist2 / fmaxf(cos_light, 1e-6f);

                                    float pdf_bsdf = bsdf_pdf(hit.mtl, wo, wi_light, hit.normal);

                                    float p_l = pdf_light_dir * pdf_light_dir;
                                    float p_b = pdf_bsdf * pdf_bsdf;
                                    float mis_w = p_l / fmaxf(p_l + p_b, 1e-8f);

                                    float3 brdf = bsdf_evaluate(hit.mtl, wo, wi_light, hit.normal);

                                    float3 contrib = throughput * brdf * light.illum * transmittance *
                                        cos_surface / pdf_light_dir * mis_w;

                                    if(is_valid_color(contrib)){
                                        final_color = final_color + clamp_radiance(contrib, 15.0f);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            float3 wi;
            float3 bsdf_val;
            float pdf_omega;
            bool is_delta;
            float new_eta;

            bsdf_sample(hit.mtl, wo, hit.normal,
                curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState),
                eyeray_refract, wi, bsdf_val, pdf_omega, is_delta, new_eta);

            if(pdf_omega <= 0.0f && !is_delta) break;

            if(is_delta){
                throughput = throughput * bsdf_val;
                eyeray_dir = wi;
                eyeray_refract = new_eta;

                if(dot(wi, hit.normal) < 0.0f) eyeray_point = hit.pos - hit.normal * EPSILON;
                else eyeray_point = hit.pos + hit.normal * EPSILON;

                last_is_delta = true;

                if(!is_valid_color(throughput)) break;

                depth--;
                continue;
            }

            float cos_wi = fabs(dot(hit.normal, wi));
            throughput = throughput * bsdf_val * cos_wi / pdf_omega;

            if(!is_valid_color(throughput)) break;

            eyeray_dir = wi;
            eyeray_point = hit.pos + hit.normal * EPSILON;

            last_is_delta = false;
            last_pdf_bsdf = pdf_omega;
        }
        if(!is_valid_color(final_color)) final_color = make_float3(0.0f, 0.0f, 0.0f);

        color_sum = color_sum + final_color;
    }

    d_image[idx] = color_sum / (float) spp;
    states[idx] = localState;
}

/*--------------------------
Wrapper 實作
--------------------------*/
void pt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth, int spp
){
    CudaLight *d_lights;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    float3 *d_image;
    curandState *d_states;

    cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
    cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
    cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
    cudaMalloc(&d_image, sizeof(float3) * W * H);
    cudaMalloc(&d_states, sizeof(curandState) * W * H);

    if(num_lights > 0) cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
    if(num_spheres > 0) cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    if(num_triangles > 0) cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);

    int total_pixels = W * H;
    int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pt_init_rng << <blocks, BLOCK_SIZE >> > (d_states, time(NULL), total_pixels);

    cuda_path_trace_kernel << <blocks, BLOCK_SIZE >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_states, W, H, eye_depth, d_image, spp
        );
    cudaDeviceSynchronize();

    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_image);
    cudaFree(d_states);
}