#include "bdpt.cuh"
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-4f

struct CudaHit{
    bool hit;
    float t;
    float3 pos;
    float3 normal;
    CudaMaterial mtl;
    int obj_id; // 除錯用
};

__device__ inline float3 operator+(const float3 &a, const float3 &b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3 &a, const float3 &b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3 &a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator*(const float3 &a, const float3 &b){ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float3 operator/(const float3 &a, float b){ return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline float dot(const float3 &a, const float3 &b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(const float3 &a, const float3 &b){ return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ inline float length(const float3 &a){ return sqrtf(dot(a, a)); }
__device__ inline float3 normalize(const float3 &a){ return a / length(a); }

__device__ inline float3 to_f3(const CudaVec3 &v){ return make_float3(v.x, v.y, v.z); }

__device__ inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

__device__ bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
    float3 oc = ro - to_f3(s.center);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.r * s.r;
    float h = b * b - c;
    if(h < 0.0f) return false;
    h = sqrtf(h);
    float t_hit = -b - h;

    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    t_hit = -b + h;
    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    return false;
}

__device__ bool intersect_triangle(const float3 &ro, const float3 &rd, const CudaTriangle &tri, float &t, float max_dist){
    float3 v0 = to_f3(tri.v0);
    float3 v1 = to_f3(tri.v1);
    float3 v2 = to_f3(tri.v2);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(rd, e2);
    float a = dot(e1, h);

    if(a > -1e-6f && a < 1e-6f) return false;

    float f = 1.0f / a;
    float3 s = ro - v0;
    float u = f * dot(s, h);

    if(u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, e1);
    float v = f * dot(rd, q);

    if(v < 0.0f || u + v > 1.0f) return false;

    float t_hit = f * dot(e2, q);

    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    return false;
}

__device__ float3 check_visibility(
    float3 p1, float3 p2,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    float3 diff = p2 - p1;
    float dist = length(diff);
    float3 dir = diff / dist;

    float3 transmission = make_float3(1.0f, 1.0f, 1.0f);

    float max_d = dist - 1e-3f;
    float min_d = 1e-3f;

    float t;
    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(p1, dir, triangles[i], t, max_d)){
            if(t > min_d){
                if(triangles[i].mtl.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

                transmission = transmission * to_f3(triangles[i].mtl.Kd);
            }
        }
    }

    for(int i = 0; i < sphere_cnt; ++i){
        const CudaSphere &s = spheres[i];

        float3 oc = p1 - to_f3(s.center);
        float b = dot(oc, dir);
        float c = dot(oc, oc) - s.r * s.r;
        float h = b * b - c;

        if(h > 0.0f){
            float sqrt_h = sqrtf(h);
            float t0 = -b - sqrt_h; 
            float t1 = -b + sqrt_h; 

            if(t0 > t1){ float temp = t0; t0 = t1; t1 = temp; }

            float enter = fmaxf(t0, min_d);
            float exit = fminf(t1, max_d);

            if(exit > enter){
                if(s.mtl.refract <= 0.0f){
                    return make_float3(0.0f, 0.0f, 0.0f);
                }

                float path_len = exit - enter;
                float3 sphere_color = to_f3(s.mtl.Kd);

                transmission = transmission * pow_f3(sphere_color, path_len * 5.f);
            }
        }
    }

    return transmission;
}

__device__ float calculate_mis_weight(
    const CudaEyeVertex *eye_path, int eye_count, int s, // s = eye vertex index (0-based)
    const CudaLightVertex *light_path, int light_count, int t, // t = light vertex index
    const float3 &dir_e_to_l, float dist2){
    CudaEyeVertex qs = eye_path[s];
    CudaLightVertex qt = light_path[t];

    float3 ns = normalize(to_f3(qs.normal));
    float3 nt = normalize(to_f3(qt.normal));

    float cos_s = fmaxf(0.0f, dot(ns, dir_e_to_l));
    float cos_t = fmaxf(0.0f, dot(nt, (dir_e_to_l * -1.0f)));

    float pdf_omega_s = cos_s / 3.14159265359f;
    float pdf_s_to_t = pdf_omega_s * cos_t / dist2; 

    float pdf_omega_t = cos_t / 3.14159265359f;
    float pdf_t_to_s = pdf_omega_t * cos_s / dist2; 
    float sum_ratios = 0.0f;

    float ratio = 1.0f;
    if(qt.pdf_fwd > 1e-6f){
        ratio = pdf_t_to_s / qt.pdf_fwd;
    }
    else{
        ratio = 0.0f;
    }

    if(t == 0){
        float p_light = qt.pdf_fwd;
        float p_bsdf = pdf_s_to_t;

        return (p_light * p_light) / (p_light * p_light + p_bsdf * p_bsdf);
    }

    float ri = 1.0f;
    for(int i = s; i >= 0; --i){
        float p_rev = eye_path[i].pdf_rev;
        float p_fwd = eye_path[i].pdf_fwd;
        ri *= (p_rev / p_fwd);
        if(i > 0) sum_ratios += ri; // 累積
    }
    return 1.0f / (1.0f + sum_ratios);
}


__global__ void eye_light_connect_kernel(
    int W, int H,
    CudaLightVertex *light_path, int light_cnt,
    CudaEyeVertex *eye_paths_flat, int *eye_offsets, int *eye_counts,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    float3 light_color,
    CudaVec3 *output,
    int connect_mode 
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int offset = eye_offsets[idx];
    int count = eye_counts[idx];

    float3 total_L = make_float3(0.0f, 0.0f, 0.0f);

    if(count == 0 || light_cnt == 0){
        output[idx] = { 0,0,0 };
        return;
    }

    for(int i = 0; i < count; ++i){
        CudaEyeVertex ev = eye_paths_flat[offset + i];
        float3 ev_pos = to_f3(ev.pos);
        float3 ev_norm = normalize(to_f3(ev.normal));
        float3 ev_tp = to_f3(ev.throughput);
        float3 ev_kd = to_f3(ev.mtl.Kd);

        float3 fE = ev_kd * (1.0f / 3.14159265359f);

        for(int j = 0; j < light_cnt; ++j){
            CudaLightVertex lv = light_path[j];
            if(connect_mode == 0 && !lv.is_light_source){
                continue;
            }

            float3 lv_pos = to_f3(lv.pos);
            float3 lv_norm = normalize(to_f3(lv.normal));
            float3 lv_tp = to_f3(lv.throughput);

            float3 d_vec = lv_pos - ev_pos;
            float dist2 = dot(d_vec, d_vec);
            if(dist2 < 1e-8f) continue;

            float dist = sqrtf(dist2);
            float3 wi = d_vec / dist;

            float cosE = fmaxf(0.0f, dot(ev_norm, wi));
            float cosL = fmaxf(0.0f, dot(lv_norm, make_float3(-wi.x, -wi.y, -wi.z)));

            if(cosE <= 0.0f || cosL <= 0.0f) continue;

            float3 transmittance = check_visibility(ev_pos + wi * 1e-3f, lv_pos, spheres, sphere_cnt, triangles, tri_cnt);

            if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                float G = (cosE * cosL) / dist2;

                float mis_w = 1.0f;
                if(connect_mode == 1){
                    mis_w = calculate_mis_weight(
                        eye_paths_flat + offset, count, i,
                        light_path, light_cnt, j,
                        d_vec, dist2
                    );
                }

                float3 contrib = ev_tp * fE * G * lv_tp * transmittance * 1.f;
                total_L = total_L + contrib;
            }
        }
    }

    output[idx].x = total_L.x;
    output[idx].y = total_L.y;
    output[idx].z = total_L.z;
}

__device__ CudaHit find_closest_hit(
    float3 ro, float3 rd,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    CudaHit best;
    best.hit = false;
    best.t = 1e20f; // Infinity

    float t;
    float max_dist = 1e20f;

    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(ro, rd, spheres[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl = spheres[i].mtl;
                best.pos = ro + rd * t;
                best.normal = normalize(best.pos - to_f3(spheres[i].center));
                if(dot(best.normal, rd) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }

    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(ro, rd, triangles[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl = triangles[i].mtl;
                best.pos = ro + rd * t;
                // Triangle 法線計算
                float3 v0 = to_f3(triangles[i].v0);
                float3 v1 = to_f3(triangles[i].v1);
                float3 v2 = to_f3(triangles[i].v2);
                best.normal = normalize(cross(v1 - v0, v2 - v0));
                if(dot(best.normal, rd) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }
    return best;
}

__device__ float3 sample_hemisphere_cosine_device(float3 N, curandState *state){
    float3 T, B;
    if(fabs(N.z) < 0.999f) T = normalize(cross(make_float3(0, 0, 1), N));
    else T = normalize(cross(make_float3(0, 1, 0), N));
    B = cross(N, T);

    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    float r = sqrtf(u1);
    float phi = 2.0f * 3.14159265359f * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    return normalize(T * x + B * y + N * z);
}

// [新增] 單位球內隨機向量 (for Glossy)
__device__ float3 random_in_unit_sphere_device(curandState *state){
    float3 p;
    do{
        p = make_float3(curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f);
    } while(dot(p, p) >= 1.0f);
    return p;
}

__device__ float3 reflect(const float3 &I, const float3 &N){
    return I - N * 2.0f * dot(N, I);
}

__global__ void init_rng_kernel(int W, int H, curandState *states, int seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void eye_tracing_kernel(
    int W, int H,
    CudaCamera cam,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    curandState *states,
    int max_depth,
    // Outputs
    CudaEyeVertex *eye_paths_flat, 
    int *eye_counts                
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int col = idx % W;
    int row = idx / W;
    
    curandState localState = states[idx]; 
    float jx = curand_uniform(&localState) - 0.5f;
    float jy = curand_uniform(&localState) - 0.5f;

    float3 pixel_pos = to_f3(cam.UL) +
        to_f3(cam.dx) * ((float) col + 0.5f + jx) +
        to_f3(cam.dy) * ((float) row + 0.5f + jy);

    float3 ray_ori = to_f3(cam.eye);
    float3 ray_dir = normalize(pixel_pos - ray_ori);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float current_refract = 1.0f;

    int depth = 0;
    int write_offset = idx * max_depth;
    
    for(depth = 0; depth < max_depth; ++depth){
        CudaHit hit = find_closest_hit(ray_ori, ray_dir, spheres, sphere_cnt, triangles, tri_cnt);

        if(!hit.hit) break; // 沒打到東西，結束路徑

        float3 N = hit.normal;
        float rn = curand_uniform(&localState);

        if(hit.mtl.reflect > 0.0f && rn <= hit.mtl.reflect){
            ray_dir = reflect(ray_dir, N);
            ray_ori = hit.pos + ray_dir * 1e-4f;
            depth--;
            continue;
        }

        if(hit.mtl.refract > 0.0f){
            float n1 = current_refract;
            float n2 = hit.mtl.refract;
            float cosNI = dot(ray_dir, N);

            float3 realN = N;
            if(cosNI > 0.0f){
                float temp = n1; n1 = n2; n2 = temp;
                realN = N * -1.0f;
            }

            float eta = n1 / n2;
            float k = 1.0f - eta * eta * (1.0f - dot(realN, ray_dir) * dot(realN, ray_dir));
            float3 T;
            if(k < 0.0f) T = make_float3(0, 0, 0); // Total Internal Reflection
            else T = ray_dir * eta - realN *(eta * dot(realN, ray_dir) + sqrtf(k));

            if(length(T) < 1e-6f){
                ray_dir = normalize(reflect(ray_dir, realN));
            }
            else{
                ray_dir = normalize(T);
                current_refract = n2;
            }
            ray_ori = hit.pos + ray_dir * 1e-4f;
            depth--;
            continue;
        }

        float diff_glos = curand_uniform(&localState);
        if(diff_glos <= hit.mtl.Kd.x){
            float3 I = normalize(ray_dir);
            float3 R = reflect(I, N);
            float3 N = R + random_in_unit_sphere_device(&localState) * (1.0f - hit.mtl.Kd.x); // 假設 Kd.x 存放 Glossy 強度

            float roughness = hit.mtl.exp > 1000 ? 0.0f : (1.0f / hit.mtl.exp * 0.05f + 0.001f); // exp 越大越光滑


            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            float3 new_dir = normalize(N + jitter);

            if(dot(new_dir, N) <= 0.0f){
                new_dir = new_dir - N * 2.0f * dot(new_dir, N) ;
            }

            new_dir = normalize(new_dir);

            ray_dir = new_dir;
            ray_ori = hit.pos + ray_dir * 1e-4f;
            depth--;
            continue;
        }

        throughput = throughput * to_f3(hit.mtl.Kd);

        CudaEyeVertex v;
        v.pos = { hit.pos.x, hit.pos.y, hit.pos.z };
        v.normal = { N.x, N.y, N.z };
        v.throughput = { throughput.x, throughput.y, throughput.z };
        v.mtl = hit.mtl; 

        eye_paths_flat[write_offset + depth] = v;

        float3 newDir = sample_hemisphere_cosine_device(N, &localState);
        ray_dir = newDir;
        ray_ori = hit.pos + ray_dir * 1e-4f;
    }

    eye_counts[idx] = depth;
    states[idx] = localState; 
}

__global__ void eye_light_connect_kernel_v2(
    int W, int H,
    CudaLightVertex *light_path, int light_cnt,
    CudaEyeVertex *eye_paths_flat, 
    int *eye_counts,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    float3 light_color,
    CudaVec3 *output,
    int connect_mode,
    int max_depth
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int count = eye_counts[idx];
    int offset = idx * max_depth; 

    float3 total_L = make_float3(0.0f, 0.0f, 0.0f);
    if(count == 0 || light_cnt == 0){
        output[idx] = { 0,0,0 };
        return;
    }

    for(int i = 0; i < count; ++i){
        CudaEyeVertex ev = eye_paths_flat[offset + i];
        float3 ev_pos = to_f3(ev.pos);
        float3 ev_norm = normalize(to_f3(ev.normal));
        float3 ev_tp = to_f3(ev.throughput);
        float3 ev_kd = to_f3(ev.mtl.Kd);
        float3 fE = ev_kd * (1.0f / 3.14159265359f);

        for(int j = 0; j < light_cnt; ++j){
            CudaLightVertex lv = light_path[j];
            if(connect_mode == 0 && !lv.is_light_source) continue;

            float3 lv_pos = to_f3(lv.pos);
            float3 d_vec = lv_pos - ev_pos;
            float dist2 = dot(d_vec, d_vec);
            if(dist2 < 1e-8f) continue;
            float dist = sqrtf(dist2);
            float3 wi = d_vec / dist;
            float3 lv_norm = normalize(to_f3(lv.normal));

            float cosE = fmaxf(0.0f, dot(ev_norm, wi));
            float cosL = fmaxf(0.0f, dot(lv_norm, wi * -1.f));
            if(cosE <= 0.0f || cosL <= 0.0f) continue;

            float3 transmittance = check_visibility(ev_pos + wi * 1e-3f, lv_pos, spheres, sphere_cnt, triangles, tri_cnt);
            if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                float G = (cosE * cosL) / dist2;
                float3 contrib = ev_tp * fE * G * to_f3(lv.throughput) * transmittance;
                total_L = total_L + contrib;
            }
        }
    }
    output[idx].x = total_L.x;
    output[idx].y = total_L.y;
    output[idx].z = total_L.z;
}


void run_cuda_pipeline(
    int W, int H,
    const CudaLightVertex *h_light_path, int light_path_size,
    const CudaSphere *h_spheres, int sphere_count,
    const CudaTriangle *h_triangles, int tri_count,
    const CudaVec3 &light_color,
    const CudaCamera &cam,
    int max_depth,
    int sample_idx,
    CudaVec3 *output_buffer,
    int connect_mode
){
    CudaLightVertex *d_light_path;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    CudaVec3 *d_output;
    curandState *d_states;
    CudaEyeVertex *d_eye_paths;
    int *d_eye_counts;

    size_t sz_lp = light_path_size * sizeof(CudaLightVertex);
    cudaMalloc(&d_light_path, sz_lp);
    cudaMalloc(&d_spheres, sphere_count * sizeof(CudaSphere));
    cudaMalloc(&d_triangles, tri_count * sizeof(CudaTriangle));
    cudaMalloc(&d_output, W * H * sizeof(CudaVec3));

    cudaMalloc(&d_eye_paths, W * H * max_depth * sizeof(CudaEyeVertex));
    cudaMalloc(&d_eye_counts, W * H * sizeof(int));
    cudaMalloc(&d_states, W * H * sizeof(curandState));

    cudaMemcpy(d_light_path, h_light_path, sz_lp, cudaMemcpyHostToDevice);
    if(sphere_count > 0) cudaMemcpy(d_spheres, h_spheres, sphere_count * sizeof(CudaSphere), cudaMemcpyHostToDevice);
    if(tri_count > 0) cudaMemcpy(d_triangles, h_triangles, tri_count * sizeof(CudaTriangle), cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    int blocks = (W * H + threads - 1) / threads;
    init_rng_kernel << <blocks, threads >> > (W, H, d_states, sample_idx + clock());

    eye_tracing_kernel << <blocks, threads >> > (
        W, H, cam,
        d_spheres, sphere_count,
        d_triangles, tri_count,
        d_states, max_depth,
        d_eye_paths, d_eye_counts
        );
    cudaDeviceSynchronize();

    float3 lc = make_float3(light_color.x, light_color.y, light_color.z);
    eye_light_connect_kernel_v2 << <blocks, threads >> > (
        W, H,
        d_light_path, light_path_size,
        d_eye_paths, // 直接使用 GPU 上的 buffer
        d_eye_counts,
        d_spheres, sphere_count,
        d_triangles, tri_count,
        lc,
        d_output,
        connect_mode,
        max_depth // stride
        );
    cudaDeviceSynchronize();

    // 6. Output
    cudaMemcpy(output_buffer, d_output, W * H * sizeof(CudaVec3), cudaMemcpyDeviceToHost);

    // 7. Cleanup
    cudaFree(d_light_path);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_output);
    cudaFree(d_eye_paths);
    cudaFree(d_eye_counts);
    cudaFree(d_states);
}

// --- Host Wrapper Implementation ---

void cuda_eye_light_connect_wrapper(
    int W, int H,
    const CudaLightVertex *h_light_path, int light_path_size,
    const CudaEyeVertex *h_eye_paths_flat,
    const int *h_eye_offsets, const int *h_eye_counts,
    const CudaSphere *h_spheres, int sphere_count,
    const CudaTriangle *h_triangles, int tri_count,
    const CudaVec3 &light_color,
    CudaVec3 *output_buffer,
    int connect_mode // <--- 傳入參數
){
    // ... (記憶體配置與複製代碼保持不變) ...
    // 1. Allocate Device Memory
    CudaLightVertex *d_light_path;
    CudaEyeVertex *d_eye_paths;
    int *d_eye_offsets, *d_eye_counts;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    CudaVec3 *d_output;

    size_t sz_lp = light_path_size * sizeof(CudaLightVertex);
    // ... (計算 sz_ep_bytes 邏輯不變) ...
    // 假設你在前面已經算好 sz_ep_bytes
    int total_eye_v = 0;
    for(int i = 0; i < W * H; i++) total_eye_v += h_eye_counts[i];
    size_t sz_ep_bytes = total_eye_v * sizeof(CudaEyeVertex);

    cudaMalloc(&d_light_path, sz_lp);
    cudaMalloc(&d_eye_paths, sz_ep_bytes);
    cudaMalloc(&d_eye_offsets, W * H * sizeof(int));
    cudaMalloc(&d_eye_counts, W * H * sizeof(int));
    cudaMalloc(&d_spheres, sphere_count * sizeof(CudaSphere));
    cudaMalloc(&d_triangles, tri_count * sizeof(CudaTriangle));
    cudaMalloc(&d_output, W * H * sizeof(CudaVec3));

    // 2. Copy Data to Device
    // ... (Copy 邏輯不變) ...
    cudaMemcpy(d_light_path, h_light_path, sz_lp, cudaMemcpyHostToDevice);
    cudaMemcpy(d_eye_paths, h_eye_paths_flat, sz_ep_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_eye_offsets, h_eye_offsets, W * H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eye_counts, h_eye_counts, W * H * sizeof(int), cudaMemcpyHostToDevice);
    if(sphere_count > 0) cudaMemcpy(d_spheres, h_spheres, sphere_count * sizeof(CudaSphere), cudaMemcpyHostToDevice);
    if(tri_count > 0) cudaMemcpy(d_triangles, h_triangles, tri_count * sizeof(CudaTriangle), cudaMemcpyHostToDevice);

    // 3. Launch Kernel
    int threads = BLOCK_SIZE;
    int blocks = (W * H + threads - 1) / threads;

    float3 lc = make_float3(light_color.x, light_color.y, light_color.z);

    eye_light_connect_kernel << <blocks, threads >> > (
        W, H,
        d_light_path, light_path_size,
        d_eye_paths, d_eye_offsets, d_eye_counts,
        d_spheres, sphere_count,
        d_triangles, tri_count,
        lc,
        d_output,
        connect_mode // <--- 傳入 Kernel
        );

    cudaDeviceSynchronize();

    // 4. Copy Result back
    cudaMemcpy(output_buffer, d_output, W * H * sizeof(CudaVec3), cudaMemcpyDeviceToHost);

    // 5. Free Memory
    cudaFree(d_light_path);
    cudaFree(d_eye_paths);
    cudaFree(d_eye_offsets);
    cudaFree(d_eye_counts);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_output);
}