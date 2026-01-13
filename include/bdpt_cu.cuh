#pragma once
#include <cuda_runtime.h>
#include "geometric.cuh"


struct CudaLightVertex{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    CudaMaterial mtl;
    bool is_light_source;
    bool is_parallel;
    float source_cutoff;
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


void bdpt_render_wrapper(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *cuda_image, int W, int H,
    int light_depth, int light_sample, int eye_depth
);