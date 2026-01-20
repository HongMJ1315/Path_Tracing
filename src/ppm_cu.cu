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

// *** 關鍵修正：加入 atomicAddVec3 實作 ***
__device__ void atomicAddVec3(CudaVec3 *addr, float3 val){
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

__device__ int get_grid_index(float3 pos, CudaVec3 min_bound, float cell_size){
    float3 rel = pos - to_f3(min_bound);
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
    CudaVec3 min_bound, float cell_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_points) return;
    if(!hit_points[idx].valid) return;

    int grid_idx = get_grid_index(to_f3(hit_points[idx].pos), min_bound, cell_size);

    int old_head = atomicExch(&hash_head[grid_idx], idx);
    hash_next[idx] = old_head;
}


// =========================================================================================
// PPM: Stage 1 - Eye Trace (Generate Hit Points)
// =========================================================================================
__global__ void ppm_eye_trace(
    const CudaSphere *spheres, int num_spheres,
    const CudaTriangle *triangles, int num_triangles,
    const CudaCamera cam,
    CudaHitPoint *hit_points,
    curandState *states,
    int W, int H, int max_depth,
    CudaVec3 *image // 為了直接寫入背景色
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];

    hit_points[idx].valid = false;
    hit_points[idx].accum_flux = { 0.0f, 0.0f, 0.0f };
    hit_points[idx].photon_count = 0;
    hit_points[idx].radius2 = PPM_RADIUS * PPM_RADIUS; // 固定半徑平方
    hit_points[idx].pixel_idx = idx;

    // Ray Generation
    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);
    float3 ray_point = to_f3(cam.eye);
    float3 pixel_pos = to_f3(cam.UL) + to_f3(cam.dx) * pixel_x + to_f3(cam.dy) * pixel_y;
    float3 ray_dir = normalize(pixel_pos - ray_point);
    float ray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(ray_point, ray_dir, spheres, num_spheres, triangles, num_triangles);

        if(!hit.hit){
            image[idx] = { 0.0f, 0.0f, 0.0f };
            break;
        }

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
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
            hit_points[idx].valid = true;
            hit_points[idx].pos = to_CudaVec3(hit.pos);
            hit_points[idx].normal = to_CudaVec3(hit.normal);
            hit_points[idx].mtl = hit.mtl;
            hit_points[idx].throughput = to_CudaVec3(throughput);
            image[idx] = { 0.0f, 0.0f, 0.0f };
            break;
        }
    }
    states[idx] = localState;
}

// =========================================================================================
// PPM: Stage 2 - Photon Trace (Scatter & Gather)
// =========================================================================================
__global__ void ppm_photon_trace(
    const CudaLight *lights, int num_lights,
    const CudaSphere *spheres, int num_spheres,
    const CudaTriangle *triangles, int num_triangles,
    CudaHitPoint *hit_points, // 被更新的目標
    int *hash_head, int *hash_next,
    CudaVec3 min_bound, CudaVec3 max_bound, float cell_size,
    curandState *states,
    int max_depth, int num_photons
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_photons) return;

    curandState localState = states[idx];

    int light_idx = idx % num_lights;
    CudaLight light = lights[light_idx];
    float3 ray_point, ray_dir;

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

    float3 photon_flux = to_f3(light.illum) * (float) num_lights;

    float ray_refract = 1.0f;

    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(ray_point, ray_dir, spheres, num_spheres, triangles, num_triangles);
        if(!hit.hit) break;

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
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
            if(hit.mtl.reflect <= 0.0f && hit.mtl.refract <= 0.0f && hit.mtl.glossy <= 0.0f){

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
                                float3 hp_pos = to_f3(hp.pos);
                                float3 diff = hp_pos - hit.pos;
                                float dist2 = dot(diff, diff);

                                if(dist2 < hp.radius2){
                                    float3 kd = to_f3(hp.mtl.Kd);
                                    float3 tp = to_f3(hp.throughput);
                                    float3 energy = photon_flux * kd * tp;
                                    atomicAddVec3(&hp.accum_flux, energy);
                                    atomicAdd(&hp.photon_count, 1.0f);
                                }
                                hp_idx = hash_next[hp_idx];
                            }
                        }
                    }
                }

                // --- SCATTER (Russian Roulette) ---
                float prob = fmaxf(hit.mtl.Kd.x, fmaxf(hit.mtl.Kd.y, hit.mtl.Kd.z));
                if(curand_uniform(&localState) > prob) break;

                photon_flux = photon_flux * to_f3(hit.mtl.Kd) / prob;
                ray_point = hit.pos + hit.normal * EPSILON;
                ray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            }
        }
    }
    states[idx] = localState;
}

// =========================================================================================
// PPM: Resolve Image
// =========================================================================================
__global__ void ppm_resolve_image(
    CudaHitPoint *hit_points,
    CudaVec3 *image,
    int W, int H,
    int total_emitted_photons
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    if(hit_points[idx].valid){
        float3 flux = to_f3(hit_points[idx].accum_flux);
        float r2 = hit_points[idx].radius2;

        // PPM Formula: Radiance = Flux / (N * PI * r^2)
        // Note: total_emitted_photons is N
        float area = PI * r2;
        float3 radiance = flux / (area * (float) total_emitted_photons);

        // Add to existing image (or overwrite if single pass)
        // Here we overwrite because the flux is accumulated in HitPoint
        image[idx] = to_CudaVec3(radiance);
    }
}


// =========================================================================================
// Host Wrapper
// =========================================================================================
void ppm_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth
){
    // 1. Setup Scene
    CudaLight *d_lights; cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
    cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);

    CudaSphere *d_spheres; cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
    cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);

    CudaTriangle *d_triangles; cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
    cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);

    // 2. Setup HitPoints & Image
    CudaHitPoint *d_hit_points;
    cudaMalloc(&d_hit_points, sizeof(CudaHitPoint) * W * H);

    CudaVec3 *d_image;
    cudaMalloc(&d_image, sizeof(CudaVec3) * W * H);

    curandState *d_states;
    light_sample = 500000;
    int max_threads = W * H > light_sample ? W * H : light_sample;
    cudaMalloc(&d_states, sizeof(curandState) * max_threads);
    ppm_init_rng << <(max_threads + 255) / 256, 256 >> > (d_states, time(NULL) + 1234, max_threads);

    // 3. Eye Trace (Generate Hit Points)
    ppm_eye_trace << <(W * H + 255) / 256, 256 >> > (
        d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_hit_points, d_states, W, H, eye_depth, d_image
        );
    cudaDeviceSynchronize();

    // 4. Build Hash Grid
    int *d_hash_head, *d_hash_next;
    cudaMalloc(&d_hash_head, sizeof(int) * HASH_TABLE_SIZE);
    cudaMalloc(&d_hash_next, sizeof(int) * W * H); // Next pointer for each HitPoint

    reset_hash_grid << <(HASH_TABLE_SIZE + 255) / 256, 256 >> >
        (d_hash_head, HASH_TABLE_SIZE);

    build_hash_grid_kernel << <(W * H + 255) / 256, 256 >> > (
        d_hit_points, W * H, d_hash_head, d_hash_next, scene_min, PPM_RADIUS
        );
    cudaDeviceSynchronize();

    // 5. Photon Trace
    int num_photons = light_sample; // Assuming light_sample passed is total photons
    ppm_photon_trace << <(num_photons + 255) / 256, 256 >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        d_hit_points, d_hash_head, d_hash_next, scene_min, scene_max, PPM_RADIUS,
        d_states, light_depth, num_photons
        );
    cudaDeviceSynchronize();

    // 6. Resolve Image
    ppm_resolve_image << <(W * H + 255) / 256, 256 >> > (
        d_hit_points, d_image, W, H, num_photons
        );
    cudaDeviceSynchronize();

    // 7. Copy Back
    cudaMemcpy(h_image, d_image, sizeof(CudaVec3) * W * H, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_lights); cudaFree(d_spheres); cudaFree(d_triangles);
    cudaFree(d_hit_points); cudaFree(d_image); cudaFree(d_states);
    cudaFree(d_hash_head); cudaFree(d_hash_next);
}