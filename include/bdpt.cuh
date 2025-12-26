#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h> 

struct CudaVec3{
    float x, y, z;
};

struct CudaMaterial{
    CudaVec3 Kd;
    CudaVec3 Kg;
    CudaVec3 Ks;
    float glossy;
    float exp;
    float refract;
    float reflect;
};

struct CudaSphere{
    CudaVec3 center;
    float r;
    CudaMaterial mtl;
    int id;
};

struct CudaTriangle{
    CudaVec3 v0, v1, v2;
    CudaMaterial mtl;
    int id;
};

struct CudaLightVertex{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    CudaMaterial mtl;
    bool is_light_source; 
    float pdf_fwd; 
    float pdf_rev; 
};

struct CudaEyeVertex{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    CudaMaterial mtl;
    float pdf_fwd; 
    float pdf_rev; 
};

struct CudaCamera{
    CudaVec3 eye;
    CudaVec3 U, V, W; // Camera 座標系的基底向量 (u, v, w) 或直接傳 UL, dx, dy
    CudaVec3 UL, dx, dy; // 我們直接用你在 main.cpp 算好的螢幕參數
};

struct CudaRay{
    CudaVec3 point, vec;
};


void cuda_eye_light_connect_wrapper(
    int W, int H,
    const CudaLightVertex *h_light_path, int light_path_size,
    const CudaEyeVertex *h_eye_paths_flat,
    const int *h_eye_offsets, const int *h_eye_counts, // 用來索引每個 pixel 有幾個頂點
    const CudaSphere *h_spheres, int sphere_count,
    const CudaTriangle *h_triangles, int tri_count,
    const CudaVec3 &light_color,
    CudaVec3 *output_buffer,
    int connect_mode // <--- 新增這個開關
);

void run_cuda_pipeline(
    int W, int H,
    const CudaLightVertex *h_light_path, int light_path_size,
    // 移除 h_eye_paths_flat, h_eye_offsets, h_eye_counts (因為現在由 GPU 產生)
    const CudaSphere *h_spheres, int sphere_count,
    const CudaTriangle *h_triangles, int tri_count,
    const CudaVec3 &light_color,
    const CudaCamera &cam, // [新增] 相機參數
    int max_depth,         // [新增] 最大深度
    int sample_idx,        // [新增] 目前的 sample index (用於亂數種子)
    CudaVec3 *output_buffer,
    int connect_mode
);