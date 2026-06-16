#include "ppm_cu.cuh"
#include <curand_kernel.h>
#include <cstdio>

#define EPSILON 1e-4f
#define PI 3.14159265358979323846f
#define BLOCK_SIZE 256


__global__ void ppm_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ void atomicAddVec3(float3 *addr, float3 val){
    atomicAdd(&(addr->x), val.x);
    atomicAdd(&(addr->y), val.y);
    atomicAdd(&(addr->z), val.z);
}

// =========================================================================================
// PPM: Hash Grid Implementation
// =========================================================================================

__device__ int hash_grid_indices(int gx, int gy, int gz){
    unsigned int h = (gx * 73856093) ^ (gy * 19349663) ^ (gz * 83492791);
    return h % HASH_TABLE_SIZE;
}

__device__ int get_grid_index(float3 pos, float3 min_bound, float cell_size){
    float3 rel = pos - min_bound;
    int gx = (int) floorf(rel.x / cell_size);
    int gy = (int) floorf(rel.y / cell_size);
    int gz = (int) floorf(rel.z / cell_size);
    return hash_grid_indices(gx, gy, gz);
}

__global__ void reset_hash_grid(int *hash_head, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) hash_head[idx] = -1;
}

__global__ void build_hash_grid_kernel(
    CudaHitPoint *hit_points, int num_points,
    int *hash_head, int *hash_next,
    float3 min_bound, float cell_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_points) return;
    if(!hit_points[idx].valid) return;

    int grid_idx = get_grid_index(hit_points[idx].pos, min_bound, cell_size);

    int old_head = atomicExch(&hash_head[grid_idx], idx);
    hash_next[idx] = old_head;
}


// =========================================================================================
// PPM: Stage 1 - Eye Trace (Generate Hit Points)
// =========================================================================================
__global__ void ppm_eye_trace(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    const CudaCamera cam,
    CudaHitPoint *hit_points,
    curandState *states,
    int W, int H, int max_depth,
    float3 *image
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];

    hit_points[idx].valid = false;
    hit_points[idx].accum_flux = make_float3(0.0f, 0.0f, 0.0f);
    hit_points[idx].photon_count = 0;
    hit_points[idx].radius2 = PPM_RADIUS * PPM_RADIUS;
    hit_points[idx].pixel_idx = idx;
    image[idx] = make_float3(0.0f, 0.0f, 0.0f);

    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);
    float3 eyeray_point = cam.eye;
    float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
    float3 eyeray_dir = normalize(pixel_pos - eyeray_point);
    float eyeray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    bool last_is_delta = true;

    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
            d_spheres, num_spheres, d_triangles, num_triangles, d_lights, num_lights);

        if(!hit.hit) break;

        float3 wo = eyeray_dir * -1.0f;

        // 直接打到光源
        if(hit.is_light){
            if(last_is_delta){
                float3 contrib = throughput * hit.mtl.base_color;
                if(is_valid_color(contrib)) image[idx] = clamp_radiance(contrib, 15.0f);
            }
            break;
        }

        float3 wi;
        float3 bsdf_val;
        float pdf_omega;
        bool is_delta;
        float new_eta;

        // BSDF 採樣
        bsdf_sample(hit.mtl, wo, hit.normal,
            curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState),
            eyeray_refract,wi, bsdf_val, pdf_omega, is_delta, new_eta);

        if(is_delta){
            // 完美折射/反射：繼續追蹤
            if(pdf_omega <= 0.0f) break;
            throughput = throughput * bsdf_val;
            eyeray_dir = wi;
            eyeray_refract = new_eta;
            eyeray_point = hit.pos + hit.normal * (dot(wi, hit.normal) < 0.0f ? -EPSILON : EPSILON);
            last_is_delta = true;
            if(!is_valid_color(throughput)) break;

            depth--;
            continue;
        }
        else{
            // 粗糙表面：建立 HitPoint 等待光子，終止射線
            hit_points[idx].valid = true;
            hit_points[idx].pos = hit.pos;
            hit_points[idx].normal = hit.normal;
            hit_points[idx].wo = wo; // 記住相機方向
            hit_points[idx].mtl = hit.mtl;
            hit_points[idx].throughput = throughput;
            break;
        }
    }
    states[idx] = localState;
}


// =========================================================================================
// PPM: Stage 2 - Photon Trace (Scatter & Gather)
// =========================================================================================
__global__ void ppm_photon_trace(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaHitPoint *hit_points,
    int *hash_head, int *hash_next,
    float3 min_bound, float3 max_bound, float cell_size,
    curandState *states,
    int max_depth, int num_photons, int spl
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_photons) return;

    curandState localState = states[idx];

    int light_idx = idx % num_lights;
    CudaLight light = d_lights[light_idx];
    float3 lightray_point, lightray_dir;
    float lightray_refract = 1.0f;

    // 光源發射邏輯 (保持不變)
    if(light.is_parallel){
        lightray_dir = normalize(light.dir);
        float3 scene_center = (min_bound + max_bound) * 0.5f;
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
        float3 local_dir = make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
        lightray_dir = normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
        lightray_point = lightray_point + lightray_dir * light.light_ball.r;
    }

    // [修正能量] 將光源總能量依據 spl 平均分配給每一顆光子
    float3 photon_flux = light.illum * (float) num_lights / fmaxf((float) spl, 1.0f);

    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(lightray_point, lightray_dir,
            d_spheres, num_spheres, d_triangles, num_triangles, d_lights, num_lights);

        if(!hit.hit || hit.is_light) break;

        float3 wi_light = lightray_dir * -1.0f; // 從交點看向光源的方向

        // --- PBR Splatting (尋找附近的 HitPoint 並注入能量) ---
        // 只有打在非完美鏡面/玻璃時，光子才會被記錄
        if(hit.mtl.eta <= 0.0f && (hit.mtl.metallic < 0.99f || hit.mtl.roughness > 0.01f)){
            int3 center_grid = make_int3(
                (int) floorf((hit.pos.x - min_bound.x) / cell_size),
                (int) floorf((hit.pos.y - min_bound.y) / cell_size),
                (int) floorf((hit.pos.z - min_bound.z) / cell_size)
            );

            for(int z = -1; z <= 1; z++){
                for(int y = -1; y <= 1; y++){
                    for(int x = -1; x <= 1; x++){
                        int neighbor_gx = center_grid.x + x;
                        int neighbor_gy = center_grid.y + y;
                        int neighbor_gz = center_grid.z + z;
                        int h_idx = hash_grid_indices(neighbor_gx, neighbor_gy, neighbor_gz);
                        int hp_idx = hash_head[h_idx];

                        while(hp_idx != -1){
                            CudaHitPoint &hp = hit_points[hp_idx];
                            // 確保法線方向一致 (避免漏光)
                            if(dot(hp.normal, hit.normal) > 0.01f){
                                float dist2 = dot(hp.pos - hit.pos, hp.pos - hit.pos);
                                if(dist2 < hp.radius2){
                                    // [PBR 核心] 評估 BRDF!
                                    // hp.wo: 相機方向, wi_light: 光子方向, hp.normal: 法線
                                    float3 brdf = bsdf_evaluate(hp.mtl, hp.wo, wi_light, hp.normal);

                                    if(is_valid_color(brdf)){
                                        float3 energy = photon_flux * brdf * hp.throughput;
                                        atomicAddVec3(&hp.accum_flux, energy);
                                        atomicAdd(&hp.photon_count, 1.0f);
                                    }
                                }
                            }
                            hp_idx = hash_next[hp_idx];
                        }
                    }
                }
            }
        }

        // --- 光子彈射 (BSDF Sample) ---
        float3 wi;
        float3 bsdf_val;
        float pdf_omega;
        bool is_delta;
        float new_eta;

        bsdf_sample(hit.mtl, wi_light, hit.normal,
            curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState),
            lightray_refract,wi, bsdf_val, pdf_omega, is_delta, new_eta);

        if(pdf_omega <= 0.0f) break;

        float cos_wi = fabs(dot(hit.normal, wi));

        if(is_delta){
            photon_flux = photon_flux * bsdf_val;
            depth--; // 完美折射/反射不扣除深度
        }
        else{
            photon_flux = photon_flux * bsdf_val * cos_wi / pdf_omega;
        }

        if(!is_valid_color(photon_flux)) break;

        lightray_dir = wi;
        lightray_refract = new_eta;
        lightray_point = hit.pos + hit.normal * (dot(wi, hit.normal) < 0.0f ? -EPSILON : EPSILON);
    }
    states[idx] = localState;
}

// =========================================================================================
// PPM: Resolve Image
// =========================================================================================
__global__ void ppm_resolve_image(
    CudaHitPoint *hit_points,
    float3 *image,
    int W, int H,
    int total_emitted_photons
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    if(hit_points[idx].valid){
        float3 flux = hit_points[idx].accum_flux;
        float r2 = hit_points[idx].radius2;

        float area = PI * r2;
        float3 radiance = flux / fmaxf(area, 1e-6f);

        if(is_valid_color(radiance)){
            // 與 PT/BDPT 相同的 Clamp 保護
            image[idx] = image[idx] + clamp_radiance(radiance, 15.0f);
        }

    }
}


// =========================================================================================
// Host Wrapper
// =========================================================================================
void ppm_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth, int spp
){
    CudaLight *d_lights; cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
    cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);

    CudaSphere *d_spheres; cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
    cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);

    CudaTriangle *d_triangles; cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
    cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);

    CudaHitPoint *d_hit_points;
    cudaMalloc(&d_hit_points, sizeof(CudaHitPoint) * W * H);

    float3 *d_image;
    cudaMalloc(&d_image, sizeof(float3) * W * H);

    // [修正] Total Photons 必須是 光源數量 * SPL (light_sample)
    int total_photons = num_lights * light_sample;

    curandState *d_states;
    int max_threads = W * H > total_photons ? W * H : total_photons;
    cudaMalloc(&d_states, sizeof(curandState) * max_threads);
    ppm_init_rng << <(max_threads + 255) / 256, 256 >> > (d_states, time(NULL) + 1234, max_threads);

    // 1. Eye Trace
    ppm_eye_trace << <(W * H + 255) / 256, 256 >> > (
        d_lights, num_lights,
        d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_hit_points, d_states, W, H, eye_depth, d_image
        );
    cudaDeviceSynchronize();

    // 2. Build Hash Grid
    int *d_hash_head, *d_hash_next;
    cudaMalloc(&d_hash_head, sizeof(int) * HASH_TABLE_SIZE);
    cudaMalloc(&d_hash_next, sizeof(int) * W * H);

    reset_hash_grid << <(HASH_TABLE_SIZE + 255) / 256, 256 >> >
        (d_hash_head, HASH_TABLE_SIZE);

    build_hash_grid_kernel << <(W * H + 255) / 256, 256 >> > (
        d_hit_points, W * H, d_hash_head, d_hash_next, scene_min, PPM_RADIUS
        );
    cudaDeviceSynchronize();

    // 3. Photon Trace
    ppm_photon_trace << <(total_photons + 255) / 256, 256 >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        d_hit_points, d_hash_head, d_hash_next, scene_min, scene_max, PPM_RADIUS,
        d_states, light_depth, total_photons, light_sample // 傳入 spl (light_sample)
        );
    cudaDeviceSynchronize();

    // 4. Resolve Image
    ppm_resolve_image << <(W * H + 255) / 256, 256 >> > (
        d_hit_points, d_image, W, H, total_photons
        );
    cudaDeviceSynchronize();

    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    cudaFree(d_lights); cudaFree(d_spheres); cudaFree(d_triangles);
    cudaFree(d_hit_points); cudaFree(d_image); cudaFree(d_states);
    cudaFree(d_hash_head); cudaFree(d_hash_next);
}