#include "bdpt.cuh"
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-4f

// --- Helper Math Functions ---
__device__ inline float3 operator+(const float3 &a, const float3 &b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3 &a, const float3 &b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3 &a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator*(const float3 &a, const float3 &b){ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float3 operator/(const float3 &a, float b){ return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline float dot(const float3 &a, const float3 &b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(const float3 &a, const float3 &b){ return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ inline float length(const float3 &a){ return sqrtf(dot(a, a)); }
__device__ inline float3 normalize(const float3 &a){ return a / length(a); }

// Convert CudaVec3 to float3
__device__ inline float3 to_f3(const CudaVec3 &v){ return make_float3(v.x, v.y, v.z); }

// --- 新增 Helper: float3 的次方運算 ---
__device__ inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

// --- Intersection Logic ---

// Sphere Intersection
__device__ bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
    float3 oc = ro - to_f3(s.center);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.r * s.r;
    float h = b * b - c;
    if(h < 0.0f) return false;
    h = sqrtf(h);
    float t_hit = -b - h;

    // Check constraints
    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    // Check second hit if inside
    t_hit = -b + h;
    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    return false;
}

// Triangle Intersection (Möller–Trumbore)
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

// Check visibility (Shadow Ray)
// Returns transmittance (1.0 = visible, 0.0 = occluded)
// 簡化版：如果有任何不透明物體阻擋則回傳 0，若是透明物體目前暫時忽略或可擴充
// Check visibility (Shadow Ray) with Volumetric Absorption
// Returns transmittance color (float3)
// 1.0 = 完全無遮擋, 0.0 = 完全遮擋, 中間值 = 半透明/有顏色衰減
__device__ float3 check_visibility(
    float3 p1, float3 p2,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    float3 diff = p2 - p1;
    float dist = length(diff);
    float3 dir = diff / dist;

    // 初始化穿透率為全白 (1.0, 1.0, 1.0)
    float3 transmission = make_float3(1.0f, 1.0f, 1.0f);

    // 稍微縮短檢測距離，避免打到 p1 或 p2 自身
    float max_d = dist - 1e-3f;
    float min_d = 1e-3f;

    // --- 1. 檢查三角形 (假設三角形為薄膜或邊界) ---
    // 注意：若場景複雜，這段 O(N) 遍歷會很慢，建議只對少數物件使用
    float t;
    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(p1, dir, triangles[i], t, max_d)){
            if(t > min_d){
                // 如果是不透明 (refract <= 0)，則完全遮擋
                if(triangles[i].mtl.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

                // 如果是透明，乘上材質顏色 (薄膜近似)
                transmission = transmission * to_f3(triangles[i].mtl.Kd);
            }
        }
    }

    // --- 2. 檢查球體 (計算體積衰減) ---
    for(int i = 0; i < sphere_cnt; ++i){
        const CudaSphere &s = spheres[i];

        // 數學推導：射線與球的交點
        float3 oc = p1 - to_f3(s.center);
        float b = dot(oc, dir);
        float c = dot(oc, oc) - s.r * s.r;
        float h = b * b - c;

        // 如果射線穿過球體 (h > 0)
        if(h > 0.0f){
            float sqrt_h = sqrtf(h);
            float t0 = -b - sqrt_h; // 進點
            float t1 = -b + sqrt_h; // 出點

            // 確保 t0 < t1
            if(t0 > t1){ float temp = t0; t0 = t1; t1 = temp; }

            // 計算射線線段 [min_d, max_d] 與 球體區間 [t0, t1] 的重疊長度
            // 重疊區間 [enter, exit]
            float enter = fmaxf(t0, min_d);
            float exit = fminf(t1, max_d);

            // 如果有重疊 (走在球體內部)
            if(exit > enter){
                // 如果是不透明球體，且射線穿過了它 -> 視為遮擋
                if(s.mtl.refract <= 0.0f){
                    return make_float3(0.0f, 0.0f, 0.0f);
                }

                // 如果是透明球體 -> 計算 Beer-Lambert 衰減
                float path_len = exit - enter;
                float3 sphere_color = to_f3(s.mtl.Kd);

                // 公式：T = Color ^ distance 
                // (假設 Kd 是單位距離的穿透色)
                transmission = transmission * pow_f3(sphere_color, path_len * 5.f);
                // * 5.0f 是密度係數，你可以調整這個數值來控制液體濃稠度
            }
        }
    }

    return transmission;
}
// --- Main Kernel ---

__global__ void eye_light_connect_kernel(
    int W, int H,
    CudaLightVertex *light_path, int light_cnt,
    CudaEyeVertex *eye_paths_flat, int *eye_offsets, int *eye_counts,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    float3 light_color,
    CudaVec3 *output,
    int connect_mode // <--- 接收開關參數
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

    // Eye Path Loop
    for(int i = 0; i < count; ++i){
        CudaEyeVertex ev = eye_paths_flat[offset + i];
        float3 ev_pos = to_f3(ev.pos);
        float3 ev_norm = normalize(to_f3(ev.normal));
        float3 ev_tp = to_f3(ev.throughput);
        float3 ev_kd = to_f3(ev.mtl.Kd);

        float3 fE = ev_kd * (1.0f / 3.14159265359f); // Lambert

        // Light Path Loop
        for(int j = 0; j < light_cnt; ++j){
            CudaLightVertex lv = light_path[j];

            // --- [關鍵修改] 模式切換邏輯 ---
            // connect_mode == 0 (Path Tracing / Next Event Estimation):
            // 僅當目標頂點是「光源本身」時才進行連接。
            // 這樣就等於只做 Direct Light Sampling (NEE)。
            if(connect_mode == 0 && !lv.is_light_source){
                continue;
            }
            // connect_mode == 1 (BDPT):
            // 不做過濾，連接所有光路上的點 (VPLs)。
            // -----------------------------

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
                float3 contrib = ev_tp * fE * G * lv_tp * transmittance;
                total_L = total_L + contrib;
            }
        }
    }

    output[idx].x = total_L.x;
    output[idx].y = total_L.y;
    output[idx].z = total_L.z;
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