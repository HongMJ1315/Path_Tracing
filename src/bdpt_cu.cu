#include "bdpt_cu.cuh"
#include <cstdio>
#include <chrono>
#define BLOCK_SIZE 256

__global__ void bdpt_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}
/*--------------------------
Light Tracing Kernel
--------------------------*/
__global__ void cuda_light_trace(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    const float3 min_bound, const float3 max_bound,
    CudaLightVertex *cuda_light_vertices,
    curandState *states,
    int max_depth,int spl, int total_paths
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_paths) return;

    int light_idx = idx % num_lights;
    CudaLight light = d_lights[light_idx];


    int path_base_idx = idx * max_depth;

    float3 lightray_point, lightray_dir;
    float ray_refract = 1.0f;


    curandState localState = states[idx];

    if(light.is_parallel){
        lightray_dir = normalize(light.dir);

        float3 scene_center = (min_bound) +max_bound * 0.5f;
        float3 diag = max_bound - min_bound;
        float scene_radius = length(diag) * 0.5f;

        float3 w = lightray_dir;
        float3 u, v;

        if(fabs(w.x) > 0.9f) u = make_float3(0.0f, 1.0f, 0.0f);
        else u = make_float3(1.0f, 0.0f, 0.0f);

        v = normalize(cross(w, u));
        u = normalize(cross(v, w));

        float r1 = curand_uniform(&localState);
        float r2 = curand_uniform(&localState);

        float plane_size = scene_radius * 2.0f;
        float offset_u = (r1 - 0.5f) * plane_size;
        float offset_v = (r2 - 0.5f) * plane_size;

        lightray_point = scene_center - lightray_dir * (scene_radius * 2.0f) + u * offset_u + v * offset_v;
    }
    else{
        lightray_point = light.pos;

        float3 w = normalize(light.dir);
        float3 u, v;

        if(fabs(w.x) > 0.9f) u = make_float3(0.0f, 1.0f, 0.0f);
        else u = make_float3(1.0f, 0.0f, 0.0f);

        v = normalize(cross(w, u));
        u = normalize(cross(v, w));

        float u1 = curand_uniform(&localState);
        float u2 = curand_uniform(&localState);
        float theta = acosf(1.0f - u1 * (1.0f - cosf(light.cutoff)));
        float phi = 2.0f * PI * u2;

        float3 local_dir = make_float3(
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        );

        lightray_dir = normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
        lightray_point = lightray_point + lightray_dir * light.light_ball.r; // 避免自相交
    }

    float3 throughput = light.illum / fmaxf((float) spl, 1.0f);
    
    CudaLightVertex &vertex0 = cuda_light_vertices[path_base_idx];
    vertex0.pos = lightray_point;
    vertex0.normal = lightray_dir;
    vertex0.throughput = throughput;
    vertex0.is_light_source = true;
    vertex0.source_cutoff = light.cutoff;
    vertex0.is_parallel = light.is_parallel;

    float3 last_normal = lightray_dir;
    float3 last_pos = lightray_point;
    float last_pdf_omega = 1.0f / PI;

    float3 wo = lightray_dir * -1.0f;
    float3 wi;
    float3 bsdf_val;
    float pdf_omega;
    bool is_delta;


    for(int depth = 1; depth < max_depth; depth++){
        CudaLightVertex &vertex = cuda_light_vertices[path_base_idx + depth];

        CudaHit hit = find_closest_hit(lightray_point, lightray_dir,
            d_spheres, num_spheres,
            d_triangles, num_triangles,
            d_lights, num_lights);
        if(!hit.hit) break;
        if(hit.is_light){
            vertex.pos = hit.pos;
            vertex.normal = hit.normal;
            vertex.throughput = throughput;
            vertex.mtl_old = hit.mtl_old;
            vertex.mtl = hit.mtl;
            vertex.is_light_source = true;
            vertex.source_cutoff = 0.0f;
            vertex.is_parallel = false;

            break;
        }
        if(length(throughput) < 1e-4f) break;

        float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
        if(dist2 < 1e-6f) break;

        float cos_at_hit = fabs(dot(hit.normal, lightray_dir * -1.0f));
        float cos_at_prev = fabs(dot(last_normal, lightray_dir));

        float pdf_fwd = last_pdf_omega * cos_at_hit / dist2;


        float3 wo = lightray_dir * -1.0f;
        float3 wi;
        float3 bsdf_val;
        float pdf_omega;
        bool is_delta;
        float new_eta;

        bsdf_sample(hit.mtl, wo, hit.normal,
            curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState),
            ray_refract,wi, bsdf_val, pdf_omega, is_delta, new_eta);

        if(pdf_omega <= 0.0f && !is_delta) break;


        if(is_delta){
            throughput = throughput * bsdf_val;
            lightray_dir = wi;
            ray_refract = new_eta;

            if(dot(wi, hit.normal) < 0.0f){
                lightray_point = hit.pos - hit.normal * EPSILON;
            }
            else{
                lightray_point = hit.pos + hit.normal * EPSILON;
            }

            depth--;
            continue;
        }


        vertex.pos = hit.pos;
        vertex.normal = hit.normal;
        vertex.throughput = throughput;
        vertex.mtl_old = hit.mtl_old;
        vertex.mtl = hit.mtl;
        vertex.is_light_source = false;

        float pdf_rev_omega = bsdf_pdf(hit.mtl, wi, wo, hit.normal);
        float pdf_rev = pdf_rev_omega * cos_at_prev / dist2; // 轉為 Area Measure

        vertex.pdf_fwd = pdf_fwd;
        vertex.pdf_rev = pdf_rev;

        float cos_wi = fabs(dot(hit.normal, wi));
        throughput = throughput * bsdf_val * cos_wi / pdf_omega;

        if(!is_valid_color(throughput)) break;

        lightray_dir = wi;
        lightray_point = hit.pos + hit.normal * EPSILON;

        last_pdf_omega = pdf_omega;
        last_normal = hit.normal;
        last_pos = hit.pos;

    }
    states[idx] = localState;
}


__device__ float calculate_mis_weight(
    const CudaEyeVertex *eye_path, int s_idx,
    const CudaLightVertex *light_path, int t_idx,
    const float3 &dir_e_to_l, float dist2,
    const CudaCamera &camera){
    if(s_idx < 0 || t_idx < 0) return 0.0f;

    const CudaEyeVertex &ev = eye_path[s_idx];
    const CudaLightVertex &lv = light_path[t_idx];

    float3 ns = normalize(ev.normal);
    float3 nt = normalize(lv.normal);

    float cos_s = fmaxf(0.0f, dot(ns, dir_e_to_l));
    float cos_t = fmaxf(0.0f, dot(nt, dir_e_to_l * -1.0f));

    if(cos_s <= 0.0f || cos_t <= 0.0f || dist2 < 1e-6f) return 0.0f;

    float3 wo_s;
    if(s_idx == 0) wo_s = normalize(camera.eye - ev.pos);
    else wo_s = normalize(eye_path[s_idx - 1].pos - ev.pos);

    float3 wo_t;
    if(t_idx == 0) wo_t = normalize(lv.normal);
    else wo_t = normalize(light_path[t_idx - 1].pos - lv.pos);

    float pdf_omega_s = bsdf_pdf(ev.mtl, wo_s, dir_e_to_l, ns);
    float pdf_omega_t = bsdf_pdf(lv.mtl, wo_t, dir_e_to_l * -1.0f, nt);

    pdf_omega_s = fmaxf(pdf_omega_s, 1e-6f);
    pdf_omega_t = fmaxf(pdf_omega_t, 1e-6f);

    float pdf_s_to_t = pdf_omega_s * cos_t / dist2;
    float pdf_t_to_s = pdf_omega_t * cos_s / dist2;
    float sum_ratios = 1.0f;

    float current_ratio = 1.0f;
    float prev_pdf_rev = pdf_t_to_s;

    for(int i = s_idx; i > 0; --i){
        const CudaEyeVertex &curr_e = eye_path[i];
        const CudaEyeVertex &prev_e = eye_path[i - 1];

        if(curr_e.mtl.eta > 0.0f || curr_e.mtl.eta > 0.0f){
            break;
        }

        current_ratio *= prev_pdf_rev / curr_e.pdf_fwd;
        sum_ratios += current_ratio;

        prev_pdf_rev = curr_e.pdf_rev;
    }

    current_ratio = 1.0f;
    prev_pdf_rev = pdf_s_to_t;

    for(int i = t_idx; i > 0; --i){
        const CudaLightVertex &curr_l = light_path[i];

        if(curr_l.is_light_source){
            current_ratio *= prev_pdf_rev / curr_l.pdf_fwd;
            sum_ratios += current_ratio;
            break;
        }

        if(curr_l.mtl.eta > 0.0f || curr_l.mtl.eta > 0.0f){
            break;
        }

        current_ratio *= prev_pdf_rev / fmaxf(curr_l.pdf_fwd, 1e-8f); // 防呆
        sum_ratios += current_ratio;
        prev_pdf_rev = curr_l.pdf_rev;
    }


    if(isnan(sum_ratios) || isinf(sum_ratios) || sum_ratios <= 0.0f){
        return 0.0f;
    }

    return 1.0f / sum_ratios;
}

/*--------------------------
Eye Trace Kernel (Fixed Logic)
--------------------------*/
__global__ void cuda_eye_trace_and_connect(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    const float3 min_bound, const float3 max_bound,
    const CudaCamera cuda_camera,
    CudaLightVertex *cuda_light_vertices, int num_light_vertices,
    CudaEyeVertex *cuda_eye_vertices,
    curandState *states,
    int W, int H,
    int max_depth, int light_path_depth, int light_sample,
    float3 *cuda_image, int spp
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    // Generate Eye Ray
    int px = idx % W;
    int py = idx / W;

    int path_base_idx = idx * max_depth;
    curandState localState = states[idx];

    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    float3 *pixel = &cuda_image[idx];


    for(int s = 0; s < spp; s++){

        float pixel_x = (float) px + curand_uniform(&localState);
        float pixel_y = (float) py + curand_uniform(&localState);

        float3 eyeray_point = cuda_camera.eye;

        float3 pixel_pos = cuda_camera.UL +
            cuda_camera.dx * pixel_x +
            cuda_camera.dy * pixel_y;

        float3 eyeray_dir = normalize(pixel_pos - cuda_camera.eye);
        float eyeray_refract = 1.0f;
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

        float3 last_normal = eyeray_dir;
        float3 last_pos = cuda_camera.eye;
        float last_pdf_omega = 1.0f;
        float last_roughness = 1.0f;


        float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

        // Connect and Trace
        for(int depth = 0; depth < max_depth; depth++){
            CudaEyeVertex &vertex = cuda_eye_vertices[path_base_idx + depth];

            CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
                d_spheres, num_spheres,
                d_triangles, num_triangles,
                d_lights, num_lights);
            if(!hit.hit) break;
            bool is_hit_light = hit.is_light;
            if(hit.is_light && depth == 0){
                vertex.pos = hit.pos;
                vertex.normal = hit.normal;
                vertex.throughput = throughput;
                vertex.mtl_old = hit.mtl_old;
                vertex.mtl = hit.mtl;
                vertex.pdf_fwd = last_pdf_omega;
                vertex.pdf_rev = 0.0f;

                final_color = final_color + hit.mtl.base_color * light_sample;
                break;
            }

            float pdf_fwd = 1.0f;
            if(depth > 0){
                float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
                float cos_at_hit = fabs(dot(hit.normal, eyeray_dir * -1.0f));
                pdf_fwd = last_pdf_omega * cos_at_hit / fmaxf(dist2, 1e-6f);
            }


            vertex.pos = hit.pos;
            vertex.normal = hit.normal;
            vertex.throughput = throughput;
            vertex.mtl_old = hit.mtl_old;
            vertex.mtl = hit.mtl;
            vertex.pdf_fwd = 0.0f; // Placeholder
            vertex.pdf_rev = 1.0f; // Placeholder




            // Connect to Light Vertices
            float3 total_L = make_float3(0.0f, 0.0f, 0.0f);
            for(int light_idx = 0; light_idx < num_light_vertices; light_idx++){
                CudaLightVertex &lv = cuda_light_vertices[light_idx];
                float3 lv_pos = lv.pos;
                float3 lv_normal = lv.normal;
                float3 lv_throughput = lv.throughput;

                CudaEyeVertex &ev = vertex;
                float3 ev_pos = ev.pos;
                float3 ev_normal = ev.normal;
                float3 ev_throughput = ev.throughput;

                float3 d_vec = lv_pos - ev_pos;
                float dist2 = dot(d_vec, d_vec);
                if(dist2 < 1e-6f) continue;

                float dist = sqrtf(dist2);
                float3 wi = d_vec / dist; // 光線從 Eye 指向 Light 的方向

                float cosE = fmaxf(0.0f, dot(ev_normal, wi));
                float cosL = fmaxf(0.0f, dot(lv_normal, wi * -1.0f));

                if(cosE <= 0.0f || cosL <= 0.0f) continue;

                if(lv.is_light_source && lv.source_cutoff > 0.0f && !lv.is_parallel){
                    int path_idx = light_idx / light_path_depth;
                    int real_light_idx = path_idx % num_lights;

                    float3 light_dir = normalize(d_lights[real_light_idx].dir);
                    float cos_theta = dot(light_dir, wi * -1.0f);
                    float cutoff_cos = cosf(lv.source_cutoff);
                    if(cos_theta < cutoff_cos) continue;
                }

                CudaEyeVertex *current_eye_path = &cuda_eye_vertices[path_base_idx];
                int path_idx = light_idx / light_path_depth;
                CudaLightVertex *current_light_path = &cuda_light_vertices[path_idx * light_path_depth];
                int current_t_idx = light_idx % light_path_depth;

                float3 wo_e = eyeray_dir * -1.0f;
                float3 fE = bsdf_evaluate(ev.mtl, wo_e, wi, ev_normal);

                float3 fL = make_float3(1.0f, 1.0f, 1.0f);
                if(!lv.is_light_source && current_t_idx > 0){
                    float3 prev_lv_pos = current_light_path[current_t_idx - 1].pos;
                    float3 wo_l = normalize(prev_lv_pos - lv_pos);

                    fL = bsdf_evaluate(lv.mtl, wo_l, wi * -1.0f, lv_normal);
                }

                if((fE.x <= 0.0f && fE.y <= 0.0f && fE.z <= 0.0f) ||
                    (fL.x <= 0.0f && fL.y <= 0.0f && fL.z <= 0.0f)){
                    continue;
                }
                float3 transmittance = check_visibility(ev_pos + ev_normal * EPSILON, lv_pos + lv_normal * EPSILON, d_spheres, num_spheres, d_triangles, num_triangles);
                if(transmittance.x <= 0.0f && transmittance.y <= 0.0f && transmittance.z <= 0.0f){
                    continue;
                }

                float G = (cosE * cosL) / fmaxf(dist2, 1e-4f);

                float mis_w = calculate_mis_weight(
                    current_eye_path, depth,
                    current_light_path, current_t_idx,
                    d_vec, dist2,
                    cuda_camera
                );

                float3 contrib = ev_throughput * fE * G * fL * lv_throughput * transmittance * mis_w;

                if(is_valid_color(contrib)){
                    contrib = clamp_radiance(contrib, 15.0f);
                    total_L = total_L + contrib;
                }
            }
            final_color = final_color + total_L;

            // Update Ray for next bounce
            float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
            if(dist2 < 1e-6f) break;

            float cos_at_hit = fabs(dot(hit.normal, eyeray_dir * -1.0f));
            float cos_at_prev = fabs(dot(last_normal, eyeray_dir));

            pdf_fwd = last_pdf_omega * cos_at_hit / dist2;


            float3 wo = eyeray_dir * -1.0f;
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

                last_pos = hit.pos;
                last_normal = hit.normal;
                last_pdf_omega = 1.0f;
                last_roughness = hit.mtl.roughness;

                depth--;
                continue;
            }


            vertex.pos = hit.pos;
            vertex.normal = hit.normal;
            vertex.throughput = throughput;
            vertex.mtl_old = hit.mtl_old;
            vertex.mtl = hit.mtl;


            float pdf_rev_omega = bsdf_pdf(hit.mtl, wi, wo, hit.normal);
            float pdf_rev = pdf_rev_omega * cos_at_prev / dist2; // 轉為 Area Measure

            vertex.pdf_fwd = pdf_fwd;
            vertex.pdf_rev = pdf_rev;

            float cos_wi = fabs(dot(hit.normal, wi));
            throughput = throughput * bsdf_val * cos_wi / pdf_omega;

            if(!is_valid_color(throughput)) break;

            eyeray_dir = wi;
            eyeray_point = hit.pos + hit.normal * EPSILON; // 非透明材質一律向外偏移

            last_pdf_omega = pdf_omega;
            last_normal = hit.normal;
            last_pos = hit.pos;
            last_roughness = hit.mtl.roughness;
        }

        if(!is_valid_color(final_color)) final_color = make_float3(0.0f, 0.0f, 0.0f);
        color_sum = color_sum + final_color;
    }

    *pixel = color_sum / (float) spp;

    states[idx] = localState;
}

void bdpt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth, int spp, int spl
){
    // ---------------------------------------------------------
    // 1. Setup Scene Data (保持不變)
    // ---------------------------------------------------------
    CudaLight *d_lights = nullptr;
    CudaSphere *d_spheres = nullptr;
    CudaTriangle *d_triangles = nullptr;

    if(num_lights > 0){
        cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights * spl);
        cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights * spl, cudaMemcpyHostToDevice);
    }
    if(num_spheres > 0){
        cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
        cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    }
    if(num_triangles > 0){
        cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
        cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);
    }

    // ---------------------------------------------------------
    // 2. Light Tracing Phase
    // ---------------------------------------------------------
    // 計算總 Light Paths 數量 (光源數 * 採樣數)
    int total_light_paths = num_lights * light_sample * spl;

    // Light Vertex Buffer 大小
    size_t total_light_vertices_size = (size_t) total_light_paths * light_depth;

    CudaLightVertex *d_cuda_light_vertices;
    cudaError_t err = cudaMalloc(&d_cuda_light_vertices, sizeof(CudaLightVertex) * total_light_vertices_size);
    if(err != cudaSuccess){ printf("Malloc LightVertices failed: %s\n", cudaGetErrorString(err)); return; }
    cudaMemset(d_cuda_light_vertices, 0, sizeof(CudaLightVertex) * total_light_vertices_size);

    // 準備 RNG States
    // 注意：這裡我們分配 "最大需求" 的空間，通常是 W*H (因為像素數量遠大於光源路徑數)
    // 這樣可以讓 Light Trace 和 Eye Trace 共用同一個 state buffer
    size_t max_threads = (size_t) W * H;
    if(total_light_paths > max_threads) max_threads = total_light_paths;

    curandState *d_states;
    cudaMalloc(&d_states, sizeof(curandState) * max_threads);

    int threads = BLOCK_SIZE;

    // --- Light Trace Launch Configuration ---
    // 使用 total_light_paths 來計算 Block 數量
    int light_blocks = (total_light_paths + threads - 1) / threads;

    // init rng for Light Trace
    bdpt_init_rng << <light_blocks, threads >> > (d_states, time(NULL) + clock(), total_light_paths);
    cudaDeviceSynchronize();

    auto light_start = std::chrono::high_resolution_clock::now();
    cuda_light_trace << <light_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max,
        d_cuda_light_vertices, d_states,
        light_depth, spl,
        total_light_paths // 傳入正確的路徑總數
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    auto light_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> light_elapsed = light_end - light_start;
    std::cout << "Light Trace Time: " << light_elapsed.count() << " seconds\n";
    if(err != cudaSuccess) printf("Light Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 3. Eye Tracing Phase
    // ---------------------------------------------------------
    CudaEyeVertex *d_cuda_eye_vertices;
    size_t total_eye_vertices_size = (size_t) W * H * eye_depth;
    err = cudaMalloc(&d_cuda_eye_vertices, sizeof(CudaEyeVertex) * total_eye_vertices_size);
    if(err != cudaSuccess){ printf("Malloc EyeVertices failed: %s\n", cudaGetErrorString(err)); return; }

    float3 *d_image;
    cudaMalloc(&d_image, sizeof(float3) * W * H);

    // --- Eye Trace Launch Configuration ---
    // 關鍵修正：這裡必須使用 W * H 來計算 Block 數量
    int total_pixels = W * H;
    int eye_blocks = (total_pixels + threads - 1) / threads;

    // Re-seed for Eye Trace (using total_pixels)
    bdpt_init_rng << <eye_blocks, threads >> > (d_states, time(NULL) + clock() + 114514, total_pixels);
    cudaDeviceSynchronize();

    // printf("Eye Trace Launch: Pixels=%d, Blocks=%d\n", total_pixels, eye_blocks);

    auto eye_start = std::chrono::high_resolution_clock::now();
    cuda_eye_trace_and_connect << <eye_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max, cuda_camera,
        d_cuda_light_vertices, total_light_vertices_size,
        d_cuda_eye_vertices, d_states,
        W, H,
        eye_depth,
        light_depth, light_sample,
        d_image, spp
        );
    cudaDeviceSynchronize();
    auto eye_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eye_elapsed = eye_end - eye_start;
    std::cout << "Eye Trace Time: " << eye_elapsed.count() << " seconds\n";
    err = cudaGetLastError();
    if(err != cudaSuccess) printf("Eye Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 4. Retrieve Data
    // ---------------------------------------------------------
    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 5. Cleanup
    // ---------------------------------------------------------
    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_cuda_light_vertices);
    cudaFree(d_cuda_eye_vertices);
    cudaFree(d_states);
    cudaFree(d_image);
}