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

// 計算 MIS Weight (Balance Heuristic)
__device__ float calculate_mis_weight(
    const CudaEyeVertex *eye_path, int eye_count, int s, // s = eye vertex index (0-based)
    const CudaLightVertex *light_path, int light_count, int t, // t = light vertex index
    const float3 &dir_e_to_l, float dist2){
    // s: Eye path 上的頂點 index (目前連接點)
    // t: Light path 上的頂點 index (目前連接點)
    // 連接邊 edge: E[s] <-> L[t]

    // 取得連接點的資料
    CudaEyeVertex qs = eye_path[s];
    CudaLightVertex qt = light_path[t];

    float3 ns = normalize(to_f3(qs.normal));
    float3 nt = normalize(to_f3(qt.normal));

    // 1. 計算連接邊的 PDF (Area Measure)
    // pdf_camera_to_light: 假如我們從 Eye 端繼續走，採樣到 Light 端點的機率
    float cos_s = fmaxf(0.0f, dot(ns, dir_e_to_l));
    float cos_t = fmaxf(0.0f, dot(nt, (dir_e_to_l * -1.0f)));

    // 假設 Diffuse Cosine Weighted
    float pdf_omega_s = cos_s / 3.14159265359f;
    float pdf_s_to_t = pdf_omega_s * cos_t / dist2; // convert to area

    float pdf_omega_t = cos_t / 3.14159265359f;
    float pdf_t_to_s = pdf_omega_t * cos_s / dist2; // convert to area

    // 2. 累加 ratios (Veach's formulation)
    // sum_ratios = sum( p_i / p_current )
    float sum_ratios = 0.0f;

    // --- 往 Eye Path 方向回溯 (s -> s-1 -> ... -> 0) ---
    float ratio = 1.0f;
    // 初始 ratio: p(連接邊逆向) / p(L端點正向)
    // 注意：這裡需要小心 qt.pdf_fwd 是否為 0 (例如光源直射)
    if(qt.pdf_fwd > 1e-6f){
        ratio = pdf_t_to_s / qt.pdf_fwd;
    }
    else{
        ratio = 0.0f;
    }

    // 如果是單純 NEE (connect_mode=0)，我們只做這一步的判斷，
    // 但全 BDPT MIS 需要迴圈

    // 這裡為了簡化，示範 NEE + BSDF 的 2-strategy MIS (最常見需求)
    // 若要做完整 BDPT MIS，需要寫兩個 while 迴圈遍歷整條路徑

    // Strategy 1: Connection (我們正在做的)
    // Strategy 2: Hitting (如果光線直接打到物體) -> 權重對應到 qt.pdf_fwd

    // 如果這是一個單純的 NEE (Eye path 長度 s+1, Light path 長度 1)
    if(t == 0){
        // NEE MIS weight:
        // weight = p_connect^2 / (p_connect^2 + p_hit^2) (Power Heuristic)
        // 或者 Balance Heuristic: 1 / (1 + p_hit / p_connect)

        // p_hit 其實就是我們假想 "如果不連接，而是 Eye path 繼續走一步打到光源" 的機率
        // 這就是 pdf_s_to_t

        // 這裡的邏輯比較抽象，針對你的程式碼，最簡單的改法：
        // 我們比較 "Explicit Connection" vs "Implicit Hit"

        // 如果是光源點 (t=0)，且是 NEE
        // implicit_pdf = pdf_s_to_t (從 Eye 採樣到 Light 的機率)
        // explicit_pdf = qt.pdf_fwd (光源本身的採樣機率，通常是 1/Area)

        // Balance Heuristic:
        // w = explicit_pdf / (explicit_pdf + implicit_pdf) ??? 
        // 不，標準 NEE 是：
        // p_light = 光源面積採樣機率 (你的 light_ray 生成機率)
        // p_bsdf  = 視線方向採樣機率 (pdf_s_to_t)

        // 假設 qt.pdf_fwd 存的是光源採樣機率 (1/Area)
        float p_light = qt.pdf_fwd;
        float p_bsdf = pdf_s_to_t;

        return (p_light * p_light) / (p_light * p_light + p_bsdf * p_bsdf);
    }

    // 若是完整 BDPT，請使用下列迴圈結構 (虛擬碼概念，需配合你的 index):
    // /*
    float ri = 1.0f;
    for(int i = s; i >= 0; --i){
        float p_rev = eye_path[i].pdf_rev;
        float p_fwd = eye_path[i].pdf_fwd;
        ri *= (p_rev / p_fwd);
        if(i > 0) sum_ratios += ri; // 累積
    }
    // 同理對 Light path...
    return 1.0f / (1.0f + sum_ratios);
    // */

    // return 1.0f; // 預設 1.0 (無 MIS)
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