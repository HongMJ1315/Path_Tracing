#pragma once
#include <vector>
#include <map>
#include <random>
#include <cuda_runtime.h>
#include "bdpt_cu.cuh" // 直接引入 CUDA 的頂點結構！

// 執行 CPU BDPT 的主函數
void run_cpu_bdpt(
    const Camera &camera,
    std::map<int, AABB> &groups,
    const std::vector<CudaLight> &lights,
    float3 *image_buffer,
    int eye_depth, int light_depth,
    int W, int H,
    int spp, int spl
);