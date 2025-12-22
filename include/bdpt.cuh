#pragma once
#include <cuda_runtime.h>

struct CudaVec3{
    float x, y, z;
};

struct CudaMaterial{
    CudaVec3 Kd;
    float refract;
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