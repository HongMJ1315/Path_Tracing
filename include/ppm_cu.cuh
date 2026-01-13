#pragma once
#include <cuda_runtime.h>
#include "geometric.cuh"

#define PPM_RADIUS 0.05f       // 固定搜尋半徑 (可依場景尺度調整)
#define HASH_TABLE_SIZE 1000003 // 質數大小的 Hash Table

void ppm_render_wrapper(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *cuda_image, int W, int H,
    int light_depth, int light_sample, int eye_depth
);