#pragma once
#include <cuda_runtime.h>


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

struct CudaLight{
    CudaVec3 pos;
    CudaVec3 dir;
    CudaVec3 illum;
    float cutoff;
    int is_parallel;
};


struct CudaHit{
    bool hit;
    float t;
    float3 pos;
    float3 normal;
    CudaMaterial mtl;
};

void bdpt_render_wrapper(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *cuda_image, int W, int H,
    int light_depth, int light_sample, int eye_depth
);