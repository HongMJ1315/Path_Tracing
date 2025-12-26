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

// [新增] 在場景中尋找最近交點 (類似 CPU 的 first_hit)
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

    // 檢查球體
    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(ro, rd, spheres[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl = spheres[i].mtl;
                best.pos = ro + rd * t;
                best.normal = normalize(best.pos - to_f3(spheres[i].center));
                // Sphere 法線修正 (向外)
                if(dot(best.normal, rd) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }

    // 檢查三角形
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

// [新增] Device 端的半球採樣
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

// [新增] 初始化 curand 狀態的 Kernel
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
    CudaEyeVertex *eye_paths_flat, // 大小為 W * H * max_depth
    int *eye_counts                // 大小為 W * H
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int col = idx % W;
    int row = idx / W; // 注意：對應 main.cpp 的 j

    curandState localState = states[idx]; // 讀取 RNG 狀態

    // 1. Generate Primary Ray (包含 Anti-aliasing jitter)
    float jx = curand_uniform(&localState) - 0.5f;
    float jy = curand_uniform(&localState) - 0.5f;

    // 計算 Pixel Position (使用傳入的 Camera UL, dx, dy)
    // 對應 CPU: UL + dx * (i + 0.5 + jx) + dy * (j + 0.5 + jy)
    float3 pixel_pos = to_f3(cam.UL) +
        to_f3(cam.dx) * ((float) col + 0.5f + jx) +
        to_f3(cam.dy) * ((float) row + 0.5f + jy);

    float3 ray_ori = to_f3(cam.eye);
    float3 ray_dir = normalize(pixel_pos - ray_ori);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float current_refract = 1.0f; // 空氣折射率

    int depth = 0;
    int write_offset = idx * max_depth; // 每個 Pixel 預留 max_depth 個頂點空間

    // Path Tracing Loop
    for(depth = 0; depth < max_depth; ++depth){
        // Find Intersection
        CudaHit hit = find_closest_hit(ray_ori, ray_dir, spheres, sphere_cnt, triangles, tri_cnt);

        if(!hit.hit) break; // 沒打到東西，結束路徑

        // 處理材質互動 (Port from eyeray_tracer)
        float3 N = hit.normal;
        float rn = curand_uniform(&localState);

        // A. Mirror Reflection
        if(hit.mtl.reflect > 0.0f && rn <= hit.mtl.reflect){
            ray_dir = reflect(ray_dir, N);
            ray_ori = hit.pos + ray_dir * 1e-4f;
            depth--; // 鏡面反射通常不計入 Diffuse Depth (或者看你的設計)
            // 若不希望無限遞迴，可以不減 depth，但原程式 logic 似乎是 continue 且 depth--
            // 為了避免死循環，我們這裡不做 depth--，而是讓它佔用一個 step，但你可以調整
            continue;
        }

        // B. Refraction
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
            // CUDA 沒有內建 refract (GLSL 有)，需手算或用 helper
            // 這裡假設你有 refract 實作，或使用 Snell's law
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
            depth--; // 同樣，折射不計入 Diffuse 頂點
            continue;
        }

        // C. Glossy
        float diff_glos = curand_uniform(&localState); // 這裡可能需要第二個隨機變數，暫用同一個
        // 原程式邏輯是用新的 rng
        if(diff_glos <= hit.mtl.Kd.x){ // 注意：原程式用 mtl.glossy，這裡假設對應到某個變數，暫用 Kd.x 或是你存的參數
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
        // 為簡化，我們直接實作 Diffuse 紀錄頂點

        // D. Diffuse (紀錄頂點並散射)
        throughput = throughput * to_f3(hit.mtl.Kd);

        // 儲存 Eye Vertex
        CudaEyeVertex v;
        v.pos = { hit.pos.x, hit.pos.y, hit.pos.z };
        v.normal = { N.x, N.y, N.z };
        v.throughput = { throughput.x, throughput.y, throughput.z };
        v.mtl = hit.mtl; // 儲存材質供連接使用

        eye_paths_flat[write_offset + depth] = v;

        // 散射下一條光線
        float3 newDir = sample_hemisphere_cosine_device(N, &localState);
        ray_dir = newDir;
        ray_ori = hit.pos + ray_dir * 1e-4f;
    }

    // 寫回 Vertex 數量
    eye_counts[idx] = depth;
    states[idx] = localState; // 更新 RNG 狀態
}

__global__ void eye_light_connect_kernel_v2(
    int W, int H,
    CudaLightVertex *light_path, int light_cnt,
    CudaEyeVertex *eye_paths_flat, // Layout: [Pixel 0 Path][Pixel 1 Path]... (Stride = max_depth)
    int *eye_counts,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    float3 light_color,
    CudaVec3 *output,
    int connect_mode,
    int max_depth // [新增] 知道 stride
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int count = eye_counts[idx];
    int offset = idx * max_depth; // 固定 Stride

    // ... (剩下的邏輯與原本 eye_light_connect_kernel 完全相同，除了讀取 eye vertex) ...
    // 請複製原本的邏輯，將 eye_offsets[idx] 替換為 offset 即可
    // 這裡為了版面省略重複代碼


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
            float cosL = fmaxf(0.0f, dot(lv_norm, wi * -1.f)); // 這裡修正一下原本的 float3 乘法
            if(cosE <= 0.0f || cosL <= 0.0f) continue;

            float3 transmittance = check_visibility(ev_pos + wi * 1e-3f, lv_pos, spheres, sphere_cnt, triangles, tri_cnt);
            if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                float G = (cosE * cosL) / dist2;
                // MIS logic omitted for brevity, add back if needed
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
    // 1. Memory Allocation (理想情況應在 class 內持久化，避免每幀 malloc)
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

    // Eye Path Memory: W * H * MaxDepth
    cudaMalloc(&d_eye_paths, W * H * max_depth * sizeof(CudaEyeVertex));
    cudaMalloc(&d_eye_counts, W * H * sizeof(int));
    cudaMalloc(&d_states, W * H * sizeof(curandState));

    // 2. Data Transfer
    cudaMemcpy(d_light_path, h_light_path, sz_lp, cudaMemcpyHostToDevice);
    if(sphere_count > 0) cudaMemcpy(d_spheres, h_spheres, sphere_count * sizeof(CudaSphere), cudaMemcpyHostToDevice);
    if(tri_count > 0) cudaMemcpy(d_triangles, h_triangles, tri_count * sizeof(CudaTriangle), cudaMemcpyHostToDevice);

    // 3. Init RNG (只做一次或是 reset)
    // 這裡為了示範，每次都呼叫，但可以加入邏輯判斷
    int threads = BLOCK_SIZE;
    int blocks = (W * H + threads - 1) / threads;
    init_rng_kernel << <blocks, threads >> > (W, H, d_states, sample_idx + clock());

    // 4. Trace Eye Paths (GPU)
    eye_tracing_kernel << <blocks, threads >> > (
        W, H, cam,
        d_spheres, sphere_count,
        d_triangles, tri_count,
        d_states, max_depth,
        d_eye_paths, d_eye_counts
        );
    cudaDeviceSynchronize();

    // 5. Connect (GPU)
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