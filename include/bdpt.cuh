#pragma once
#include <cuda_runtime.h>

// 定義與 glm::vec3 記憶體佈局相同的結構
struct CudaVec3{
    float x, y, z;
};

struct CudaMaterial{
    CudaVec3 Kd;
    float refract;
    // Shadow ray 只需要知道是否透明或 Kd 即可
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

// 用來儲存扁平化的光路徑頂點
struct CudaLightVertex{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    // 我們不存 Object*，改存 Material 屬性，因為連線時只需要知道材質特性
    CudaMaterial mtl;
    bool is_light_source; // 標記是否為光源本身
};

// 用來儲存扁平化的眼路徑頂點
struct CudaEyeVertex{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    CudaMaterial mtl;
};

// CUDA 函式入口
// output_buffer: 存放結果顏色 (W * H)
void cuda_eye_light_connect_wrapper(
    int W, int H,
    const CudaLightVertex *h_light_path, int light_path_size,
    const CudaEyeVertex *h_eye_paths_flat,
    const int *h_eye_offsets, const int *h_eye_counts, // 用來索引每個 pixel 有幾個頂點
    const CudaSphere *h_spheres, int sphere_count,
    const CudaTriangle *h_triangles, int tri_count,
    const CudaVec3 &light_color,
    CudaVec3 *output_buffer
);