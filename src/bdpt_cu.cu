#include "bdpt_cu.cuh"
#include <cstdio>
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
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    const CudaVec3 min_bound, const CudaVec3 max_bound,
    CudaLightVertex *cuda_light_vertices,
    curandState *states,
    int max_depth, int total_paths
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_paths) return;

    int light_idx = idx % num_lights;
    CudaLight light = cuda_lights[light_idx];


    int path_base_idx = idx * max_depth;

    float3 ray_point, ray_dir;
    float ray_refract = 1.0f;
    curandState localState = states[idx];

    if(light.is_parallel){
        ray_dir = normalize(to_f3(light.dir));

        float3 scene_center = (to_f3(min_bound) + to_f3(max_bound)) * 0.5f;
        float3 diag = to_f3(max_bound) - to_f3(min_bound);
        float scene_radius = length(diag) * 0.5f;

        float3 w = ray_dir;
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

        ray_point = scene_center - ray_dir * (scene_radius * 2.0f) + u * offset_u + v * offset_v;
    }
    else{
        ray_point = to_f3(light.pos);

        float3 w = normalize(to_f3(light.dir));
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

        ray_dir = normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
    }

    float3 throughput = to_f3(light.illum);

    CudaLightVertex &vertex0 = cuda_light_vertices[path_base_idx];
    vertex0.pos = to_CudaVec3(ray_point);
    vertex0.normal = to_CudaVec3(ray_dir);
    vertex0.throughput = to_CudaVec3(throughput);
    vertex0.is_light_source = true;
    vertex0.source_cutoff = light.cutoff;
    vertex0.is_parallel = light.is_parallel;

    float3 last_normal = ray_dir;
    float3 last_pos = ray_point;
    float last_pdf_omega = 1.0f / PI;


    for(int depth = 1; depth < max_depth; depth++){
        CudaLightVertex &vertex = cuda_light_vertices[path_base_idx + depth];

        CudaHit hit = find_closest_hit(ray_point, ray_dir, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
        if(!hit.hit) break;
        if(length(throughput) < 1e-4f) break;

        float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
        float dist = sqrtf(dist2);
        if(dist2 < 1e-6f) break;

        float cos_at_hit = abs(dot(hit.normal, ray_dir * -1.0f));
        float cos_at_prev = abs(dot(last_normal, ray_dir));

        float pdf_fwd = last_pdf_omega * cos_at_hit / dist2;
        float pdf_rev_omega = cos_at_hit / PI;
        float pdf_rev = pdf_rev_omega * cos_at_prev / dist2;

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
            depth--;
            continue;
        }
        if(hit.mtl.refract > 0.0f){
            float3 refracted_dir;
            float3 I = ray_dir, N = hit.normal;
            float n1 = ray_refract;
            float n2 = hit.mtl.refract;

            float cosNI = dot(I, N);
            if(cosNI > 0.0f){
                swap(n1, n2);
                N = N * -1.0f;
                cosNI = dot(I, N);
            }
            float eta = n1 / n2;
            refracted_dir = refract(I, N, eta);
            if(length(refracted_dir) > 0.0f){
                ray_point = hit.pos - hit.normal * EPSILON;
                ray_dir = refracted_dir;
                ray_refract = hit.mtl.refract;
            }
            else{
                ray_point = hit.pos + hit.normal * EPSILON;
                ray_dir = reflect(ray_dir, hit.normal);
            }
            depth--;
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl.glossy){
            float3 perfect_reflect = reflect(ray_dir, hit.normal);
            float roughness = (hit.mtl.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl.exp * 0.0005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            ray_dir = normalize(perfect_reflect + jitter);
            if(dot(ray_dir, hit.normal) < 0.0f){
                ray_dir = ray_dir - hit.normal * dot(ray_dir, hit.normal) * 2.0f;
                ray_dir = normalize(ray_dir);
            }
            ray_point = hit.pos + ray_dir * EPSILON;
        }
        else{
            ray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            ray_point = hit.pos + ray_dir * EPSILON;
            throughput = throughput * to_f3(hit.mtl.Kd); // Diffuse attenuation

            vertex.pos = to_CudaVec3(hit.pos);
            vertex.normal = to_CudaVec3(hit.normal);
            vertex.throughput = to_CudaVec3(throughput);
            vertex.mtl = hit.mtl;
            vertex.is_light_source = false;
            vertex.pdf_fwd = pdf_fwd;
            vertex.pdf_rev = pdf_rev;

            last_pdf_omega = pdf_rev_omega;
            last_normal = hit.normal;
            last_pos = hit.pos;
        }
    }
    states[idx] = localState;
}


/*--------------------------
Eye Trace Kernel (Fixed Logic)
--------------------------*/
__global__ void cuda_eye_trace_and_connect(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    const CudaVec3 min_bound, const CudaVec3 max_bound,
    const CudaCamera cuda_camera,
    CudaLightVertex *cuda_light_vertices, int num_light_vertices,
    CudaEyeVertex *cuda_eye_vertices,
    curandState *states,
    int W, int H,
    int max_depth, int light_path_depth, CudaVec3 *cuda_image
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    // Generate Eye Ray
    int px = idx % W;
    int py = idx / W;

    int path_base_idx = idx * max_depth;
    curandState localState = states[idx];

    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);

    float3 ray_point = to_f3(cuda_camera.eye);

    float3 pixel_pos = to_f3(cuda_camera.UL) +
        to_f3(cuda_camera.dx) * pixel_x +
        to_f3(cuda_camera.dy) * pixel_y;

    float3 ray_dir = normalize(pixel_pos - to_f3(cuda_camera.eye));
    float ray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    CudaVec3 *pixel = &cuda_image[idx];

    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

    // Connect and Trace
    for(int depth = 0; depth < max_depth; depth++){
        CudaEyeVertex &vertex = cuda_eye_vertices[path_base_idx + depth];

        CudaHit hit = find_closest_hit(ray_point, ray_dir, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
        if(!hit.hit) break;

        vertex.pos = to_CudaVec3(hit.pos);
        vertex.normal = to_CudaVec3(hit.normal);
        vertex.throughput = to_CudaVec3(throughput);
        vertex.mtl = hit.mtl;
        vertex.pdf_fwd = 1.0f; // Placeholder
        vertex.pdf_rev = 1.0f; // Placeholder

        // Connect to Light Vertices
        float3 total_L = make_float3(0.0f, 0.0f, 0.0f);
        for(int light_idx = 0; light_idx < num_light_vertices; light_idx++){
            CudaLightVertex lv = cuda_light_vertices[light_idx];
            float3 lv_pos = to_f3(lv.pos);
            float3 lv_normal = to_f3(lv.normal);
            float3 lv_throughput = to_f3(lv.throughput);
            if(lv.is_light_source){
                float scene_radius = 0.075f;
                float3 w = lv_normal;
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
                lv_pos = lv_pos + u * offset_u + v * offset_v;
            }


            CudaEyeVertex &ev = vertex;
            float3 ev_pos = to_f3(ev.pos);
            float3 ev_normal = to_f3(ev.normal);
            float3 ev_throughput = to_f3(ev.throughput);
            float3 ev_kd = to_f3(ev.mtl.Kd);

            float3 fE = ev_kd / PI; // Diffuse BRDF

            float3 d_vec = lv_pos - ev_pos;
            float dist2 = dot(d_vec, d_vec);
            if(dist2 < 1e-6f) continue;

            float dist = sqrtf(dist2);
            float3 wi = d_vec / dist;

            float cosE = fmaxf(0.0f, dot(ev_normal, wi));
            float cosL = fmaxf(0.0f, dot(lv_normal, wi * -1.0f));

            if(cosE <= 0.0f || cosL <= 0.0f) continue;
            if(lv.is_light_source && lv.source_cutoff > 0.0f && !lv.is_parallel){
                // spot light cutoff test
                int path_idx = light_idx / light_path_depth;
                int real_light_idx = path_idx % num_lights;

                float3 light_dir = normalize(to_f3(cuda_lights[real_light_idx].dir));
                float cos_theta = dot(light_dir, wi * -1.0f);
                float cutoff_cos = cosf(lv.source_cutoff);
                if(cos_theta < cutoff_cos) continue;
            }
            float3 transmittance = check_visibility(ev_pos + ev_normal * EPSILON, lv_pos + lv_normal * EPSILON, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
            if(transmittance.x <= 0.0f && transmittance.y <= 0.0f && transmittance.z <= 0.0f) transmittance = make_float3(0.0f, 0.0f, 0.0f);
            float G = (cosE * cosL) / dist2;

            float3 contrib = ev_throughput * fE * G * lv_throughput * transmittance;
            total_L = total_L + contrib;
        }
        final_color = final_color + total_L;

        // Update Ray for next bounce
        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
            depth--;
            continue;
        }
        if(hit.mtl.refract > 0.0f){
            float3 refracted_dir;
            float3 I = ray_dir, N = hit.normal;
            float n1 = ray_refract;
            float n2 = hit.mtl.refract;

            float cosNI = dot(I, N);
            if(cosNI > 0.0f){
                swap(n1, n2);
                N = N * -1.0f;
                cosNI = dot(I, N);
            }
            float eta = n1 / n2;
            refracted_dir = refract(I, N, eta);
            if(length(refracted_dir) > 0.0f){
                ray_point = hit.pos - hit.normal * EPSILON;
                ray_dir = refracted_dir;
                ray_refract = hit.mtl.refract;
            }
            else{
                ray_point = hit.pos + hit.normal * EPSILON;
                ray_dir = reflect(ray_dir, hit.normal);
            }
            depth--;
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl.glossy){
            float3 perfect_reflect = reflect(ray_dir, hit.normal);
            float roughness = (hit.mtl.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl.exp * 0.0005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            ray_dir = normalize(perfect_reflect + jitter);
            if(dot(ray_dir, hit.normal) < 0.0f){
                ray_dir = ray_dir - hit.normal * dot(ray_dir, hit.normal) * 2.0f;
                ray_dir = normalize(ray_dir);
            }
            ray_point = hit.pos + ray_dir * EPSILON;
        }
        else{
            ray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            ray_point = hit.pos + ray_dir * EPSILON;
            throughput = throughput * to_f3(hit.mtl.Kd); // Diffuse attenuation
        }

    }

    *pixel = to_CudaVec3(final_color);
    states[idx] = localState;
}

void bdpt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth
){
    // ---------------------------------------------------------
    // 1. Setup Scene Data (保持不變)
    // ---------------------------------------------------------
    CudaLight *d_lights = nullptr;
    CudaSphere *d_spheres = nullptr;
    CudaTriangle *d_triangles = nullptr;

    if(num_lights > 0){
        cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
        cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
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
    int total_light_paths = num_lights * light_sample;

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

    cuda_light_trace << <light_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max,
        d_cuda_light_vertices, d_states,
        light_depth,
        total_light_paths // 傳入正確的路徑總數
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess) printf("Light Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 3. Eye Tracing Phase
    // ---------------------------------------------------------
    CudaEyeVertex *d_cuda_eye_vertices;
    size_t total_eye_vertices_size = (size_t) W * H * eye_depth;
    err = cudaMalloc(&d_cuda_eye_vertices, sizeof(CudaEyeVertex) * total_eye_vertices_size);
    if(err != cudaSuccess){ printf("Malloc EyeVertices failed: %s\n", cudaGetErrorString(err)); return; }

    CudaVec3 *d_image;
    cudaMalloc(&d_image, sizeof(CudaVec3) * W * H);

    // --- Eye Trace Launch Configuration ---
    // 關鍵修正：這裡必須使用 W * H 來計算 Block 數量
    int total_pixels = W * H;
    int eye_blocks = (total_pixels + threads - 1) / threads;

    // Re-seed for Eye Trace (using total_pixels)
    bdpt_init_rng << <eye_blocks, threads >> > (d_states, time(NULL) + 9999, total_pixels);
    cudaDeviceSynchronize();

    // printf("Eye Trace Launch: Pixels=%d, Blocks=%d\n", total_pixels, eye_blocks);

    cuda_eye_trace_and_connect << <eye_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max, cuda_camera,
        d_cuda_light_vertices, total_light_vertices_size,
        d_cuda_eye_vertices, d_states,
        W, H,
        eye_depth,
        light_depth,
        d_image
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess) printf("Eye Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 4. Retrieve Data
    // ---------------------------------------------------------
    cudaMemcpy(h_image, d_image, sizeof(CudaVec3) * W * H, cudaMemcpyDeviceToHost);

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