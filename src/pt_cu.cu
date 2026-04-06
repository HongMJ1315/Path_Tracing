#include "pt_cu.cuh"
#include <cstdio>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-4f

// 初始化隨機數生成器
__global__ void pt_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}



/*--------------------------
傳統 Path Tracing Kernel
--------------------------*/
__global__ void cuda_path_trace_kernel(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaCamera cam, curandState *states,
    int W, int H, int max_depth, float3 *d_image
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];


    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    // Ray Generation
    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);
    float3 eyeray_point = cam.eye;
    float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
    float3 eyeray_dir = normalize(pixel_pos - eyeray_point);
    float eyeray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    bool last_specular = true;

    for(int depth = 0; depth < max_depth; ++depth){
        CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
            d_spheres, num_spheres,
            d_triangles, num_triangles,
            d_lights, num_lights);
        if(!hit.hit) break;
        if(hit.is_light){
            if(last_specular){
                final_color = final_color + throughput * hit.mtl_old.Kd;
            }
            break;
        }
        float do_reflect = curand_uniform(&localState);
        if(hit.mtl_old.reflect > 0.0f && do_reflect < hit.mtl_old.reflect){
            eyeray_point = hit.pos + hit.normal * EPSILON;
            eyeray_dir = reflect(eyeray_dir, hit.normal);
            depth--;
            last_specular = true;
            continue;
        }
        if(hit.mtl_old.refract > 0.0f){
            float3 refracted_dir;
            float3 I = eyeray_dir, N = hit.normal;
            float n1 = eyeray_refract;
            float n2 = hit.mtl_old.refract;

            float cosNI = dot(I, N);
            if(cosNI > 0.0f){
                swap(n1, n2);
                N = N * -1.0f;
                cosNI = dot(I, N);
            }
            float eta = n1 / n2;
            refracted_dir = refract(I, N, eta);
            if(length(refracted_dir) > 0.0f){
                eyeray_point = hit.pos - hit.normal * EPSILON;
                eyeray_dir = refracted_dir;
                eyeray_refract = hit.mtl_old.refract;
            }
            else{
                eyeray_point = hit.pos + hit.normal * EPSILON;
                eyeray_dir = reflect(eyeray_dir, hit.normal);
            }
            depth--;
            last_specular = true;
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl_old.glossy){
            float3 perfect_reflect = reflect(eyeray_dir, hit.normal);
            float roughness = (hit.mtl_old.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl_old.exp * 0.0005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            eyeray_dir = normalize(perfect_reflect + jitter);
            if(dot(eyeray_dir, hit.normal) < 0.0f){
                eyeray_dir = eyeray_dir - hit.normal * dot(eyeray_dir, hit.normal) * 2.0f;
                eyeray_dir = normalize(eyeray_dir);
            }
            eyeray_point = hit.pos + eyeray_dir * EPSILON;
        }
        else{
            // ==========================================
            // Next Event Estimation
            // ==========================================
            if(num_lights > 0){
                int l_idx = min((int) (curand_uniform(&localState) * num_lights), num_lights - 1);
                CudaLight light = d_lights[l_idx];

                float3 light_dir;
                float light_dist;
                float3 light_color = light.illum;
                bool valid_light = true;

                if(light.is_parallel){
                    light_dir = normalize(light.dir * -1.0f);
                    light_dist = 1e9f;
                }
                else{
                    float3 light_pos = light.pos + random_in_unit_sphere_device(&localState) * light.light_ball.r; // 採樣光球表面 

                    float3 d_vec = light_pos - hit.pos;
                    float dist2 = dot(d_vec, d_vec);
                    light_dist = sqrtf(dist2);
                    light_dir = d_vec / light_dist;

                    light_color = light_color / fmaxf(dist2, 0.0001f);

                    if(light.cutoff > 0.0f){
                        float cos_l = dot(normalize(light.dir), light_dir * -1.0f);
                        if(cos_l < cosf(light.cutoff)) valid_light = false;
                    }
                }

                float cos_theta = dot(hit.normal, light_dir);

                if(valid_light && cos_theta > 0.0f){
                    float3 shadow_origin = hit.pos + hit.normal * EPSILON;
                    CudaHit shadow_hit = find_closest_hit(shadow_origin, light_dir,
                        d_spheres, num_spheres,
                        d_triangles, num_triangles,
                        d_lights, num_lights);

                    bool in_shadow = false;
                    if(shadow_hit.hit){
                        float s_dist = length(shadow_hit.pos - shadow_origin);
                        if(s_dist < light_dist - 0.01f && !shadow_hit.is_light){
                            in_shadow = true;
                        }
                    }

                    if(!in_shadow){
                        float3 brdf = hit.mtl_old.Kd / 3.14159265359f;
                        final_color = final_color + throughput * brdf * light_color * cos_theta * (float) num_lights;
                    }
                }
            }

            eyeray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            eyeray_point = hit.pos + eyeray_dir * EPSILON;
            throughput = throughput * hit.mtl_old.Kd;

            last_specular = false;
        }
    }

    d_image[idx] = final_color;
    states[idx] = localState;
}

/*--------------------------
Wrapper 實作 (保持與 BDPT 介面完全一致)
--------------------------*/
void pt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth
){
    // 1. 分配 GPU 記憶體 (與 bdpt_cu.cu 邏輯一致)
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

    cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);

    // 2. 初始化隨機數
    int total_pixels = W * H;
    int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pt_init_rng << <blocks, BLOCK_SIZE >> > (d_states, time(NULL), total_pixels);

    // 3. 執行 Path Tracing (這裡將 eye_depth 作為 PT 的 max_depth)
    cuda_path_trace_kernel << <blocks, BLOCK_SIZE >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_states, W, H, eye_depth, d_image
        );
    cudaDeviceSynchronize();

    // 4. 回傳數據
    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    // 5. 釋放資源
    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_image);
    cudaFree(d_states);
}