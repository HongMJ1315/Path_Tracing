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
__device__ float check_visibility(
    float3 p1, float3 p2,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    float3 diff = p2 - p1;
    float dist = length(diff);
    float3 dir = diff / dist;

    float t;
    float max_d = dist - 1e-3f; // 稍微縮短避免打到目標點自己

    // 檢查所有球體
    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(p1, dir, spheres[i], t, max_d)){
            // 如果是不透明的 (refract <= 0)，則視為遮擋
            if(spheres[i].mtl.refract <= 0.0f) return 0.0f;
        }
    }

    // 檢查所有三角形
    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(p1, dir, triangles[i], t, max_d)){
            if(triangles[i].mtl.refract <= 0.0f) return 0.0f;
        }
    }

    return 1.0f;
}

// --- Main Kernel ---

__global__ void eye_light_connect_kernel(
    int W, int H,
    CudaLightVertex *light_path, int light_cnt,
    CudaEyeVertex *eye_paths_flat, int *eye_offsets, int *eye_counts,
    CudaSphere *spheres, int sphere_cnt,
    CudaTriangle *triangles, int tri_cnt,
    float3 light_color,
    CudaVec3 *output
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    // 取得該 Pixel 對應的 Eye Path 資訊
    int offset = eye_offsets[idx];
    int count = eye_counts[idx];

    float3 total_L = make_float3(0.0f, 0.0f, 0.0f);

    if(count == 0 || light_cnt == 0){
        output[idx] = { 0,0,0 };
        return;
    }

    // 雙向連接迴圈：Eye Path 的每個點 <--> Light Path 的每個點
    for(int i = 0; i < count; ++i){
        CudaEyeVertex ev = eye_paths_flat[offset + i];
        float3 ev_pos = to_f3(ev.pos);
        float3 ev_norm = normalize(to_f3(ev.normal));
        float3 ev_tp = to_f3(ev.throughput);
        float3 ev_kd = to_f3(ev.mtl.Kd);

        float3 fE = ev_kd * (1.0f / 3.14159265359f); // Lambert

        for(int j = 0; j < light_cnt; ++j){
            CudaLightVertex lv = light_path[j];
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

            // Visibility Check
            float transmittance = check_visibility(ev_pos + wi * 1e-3f, lv_pos, spheres, sphere_cnt, triangles, tri_cnt);

            if(transmittance > 0.0f){
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
    CudaVec3 *output_buffer
){
    // 1. Allocate Device Memory
    CudaLightVertex *d_light_path;
    CudaEyeVertex *d_eye_paths;
    int *d_eye_offsets, *d_eye_counts;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    CudaVec3 *d_output;

    size_t sz_lp = light_path_size * sizeof(CudaLightVertex);
    size_t sz_ep = h_eye_offsets[W * H - 1] + h_eye_counts[W * H - 1]; // 總 EyeVertex 數量 (估算，嚴謹應傳入總數)
    // 修正：應從外部傳入 total_eye_vertices 數量，此處假設呼叫者知道如何傳遞，
    // 為簡化，我們用 offsets 最後一個元素的 index + count 來計算總大小
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
        d_output
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