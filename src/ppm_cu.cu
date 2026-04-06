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
    float3 *image // 為了直接寫入背景色
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
    float3 eyeray_point = cam.eye;
    float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
    float3 eyeray_dir = normalize(pixel_pos - eyeray_point);
    float eyeray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
            d_spheres, num_spheres,
            d_triangles, num_triangles,
            d_lights, num_lights);
        if(!hit.hit){
            image[idx] = { 0.0f, 0.0f, 0.0f };
            break;
        }
        if(hit.is_light){
            image[idx] = hit.mtl_old.Kd; // 直接將光源顏色寫入背景
            break;
        }

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl_old.reflect > 0.0f && do_reflect < hit.mtl_old.reflect){
            eyeray_point = hit.pos + hit.normal * EPSILON;
            eyeray_dir = reflect(eyeray_dir, hit.normal);
            depth--;
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
            hit_points[idx].valid = true;
            hit_points[idx].pos = hit.pos;
            hit_points[idx].normal = hit.normal;
            hit_points[idx].mtl_old = hit.mtl_old;
            hit_points[idx].throughput = throughput;
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
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaHitPoint *hit_points, // 被更新的目標
    int *hash_head, int *hash_next,
    float3 min_bound, float3 max_bound, float cell_size,
    curandState *states,
    int max_depth, int num_photons
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_photons) return;

    curandState localState = states[idx];

    int light_idx = idx % num_lights;
    CudaLight light = d_lights[light_idx];
    float3 lightray_point, lightray_dir;
    float lightray_refract = 1.0f;

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

        float3 local_dir = make_float3(
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        );

        lightray_dir = normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
        lightray_point = lightray_point + lightray_dir * light.light_ball.r; // 避免自相交
    }

    float3 photon_flux = light.illum * (float) num_lights;


    for(int depth = 0; depth < max_depth; depth++){
        CudaHit hit = find_closest_hit(lightray_point, lightray_dir,
            d_spheres, num_spheres,
            d_triangles, num_triangles,
            d_lights, num_lights);
        if(!hit.hit) break;
        if(hit.is_light) break; // 光源不反射

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl_old.reflect > 0.0f && do_reflect < hit.mtl_old.reflect){
            lightray_point = hit.pos + hit.normal * EPSILON;
            lightray_dir = reflect(lightray_dir, hit.normal);
            depth--;
            continue;
        }
        if(hit.mtl_old.refract > 0.0f){
            float3 refracted_dir;
            float3 I = lightray_dir, N = hit.normal;
            float n1 = lightray_refract;
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
                lightray_point = hit.pos - hit.normal * EPSILON;
                lightray_dir = refracted_dir;
                lightray_refract = hit.mtl_old.refract;
            }
            else{
                lightray_point = hit.pos + hit.normal * EPSILON;
                lightray_dir = reflect(lightray_dir, hit.normal);
            }
            depth--;
            continue;
        }

        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl_old.glossy){
            float3 perfect_reflect = reflect(lightray_dir, hit.normal);
            float roughness = (hit.mtl_old.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl_old.exp * 0.0005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            lightray_dir = normalize(perfect_reflect + jitter);
            if(dot(lightray_dir, hit.normal) < 0.0f){
                lightray_dir = lightray_dir - hit.normal * dot(lightray_dir, hit.normal) * 2.0f;
                lightray_dir = normalize(lightray_dir);
            }
            lightray_point = hit.pos + lightray_dir * EPSILON;
        }
        else{

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

                            if(dot(hp.normal, hit.normal) > 0.9f){
                                float3 hp_pos = hp.pos;
                                float3 diff = hp_pos - hit.pos;
                                float dist2 = dot(diff, diff);

                                if(dist2 < hp.radius2){
                                    float3 kd = hp.mtl_old.Kd;
                                    float3 tp = hp.throughput;
                                    float3 energy = photon_flux * kd * tp;
                                    atomicAddVec3(&hp.accum_flux, energy);
                                    atomicAdd(&hp.photon_count, 1.0f);
                                }
                            }
                            hp_idx = hash_next[hp_idx];
                        }
                    }
                }
            }


            lightray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            photon_flux = photon_flux * hit.mtl_old.Kd;
            lightray_point = hit.pos + lightray_dir * EPSILON;
        }
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
        float3 radiance = flux / (area * (float) total_emitted_photons);

        image[idx] = radiance;
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

    float3 *d_image;
    cudaMalloc(&d_image, sizeof(float3) * W * H);

    curandState *d_states;
    int max_threads = W * H > light_sample ? W * H : light_sample;
    cudaMalloc(&d_states, sizeof(curandState) * max_threads);
    ppm_init_rng << <(max_threads + 255) / 256, 256 >> > (d_states, time(NULL) + 1234, max_threads);

    // 3. Eye Trace (Generate Hit Points)
    ppm_eye_trace << <(W * H + 255) / 256, 256 >> > (
        d_lights, num_lights,
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
    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_lights); cudaFree(d_spheres); cudaFree(d_triangles);
    cudaFree(d_hit_points); cudaFree(d_image); cudaFree(d_states);
    cudaFree(d_hash_head); cudaFree(d_hash_next);
}