#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "object.h"

#define EPSILON 1e-4f
#define PI 3.14159265358979323846f

struct CudaVec3{ float x, y, z; };

struct CudaMaterial{
    CudaVec3 Kd, Kg, Ks;
    float glossy, exp, refract, reflect;
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

struct CudaHit{
    bool hit;
    float t;
    float3 pos, normal;
    CudaMaterial mtl;
};

struct CudaHitPoint{
    CudaVec3 pos;
    CudaVec3 normal;
    CudaVec3 throughput;
    CudaVec3 accum_flux;
    CudaMaterial mtl;
    int pixel_idx;
    float radius2;
    float photon_count;
    bool valid;
};

struct CudaCamera{
    CudaVec3 eye, U, V, W, UL, dx, dy;
};

struct CudaRay{ CudaVec3 point, vec; };

struct CudaLight{
    CudaVec3 pos, dir, illum;
    float cutoff;
    int is_parallel;
};

// Host Helper Declarations
CudaVec3 to_cv3(const glm::vec3 &v);
CudaMaterial to_cmtl(const Material &m);
CudaVec3 normalize_cuda(const CudaVec3 &v);

// =========================================================
// Device Inline Implementations (必須放在 Header)
// =========================================================

__device__ inline float3 operator+(const float3 &a, const float3 &b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3 &a, const float3 &b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3 &a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator*(const float3 &a, const float3 &b){ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline float3 operator/(const float3 &a, float b){ return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline float dot(const float3 &a, const float3 &b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(const float3 &a, const float3 &b){ return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ inline float length(const float3 &a){ return sqrtf(dot(a, a)); }
__device__ inline float3 normalize(const float3 &a){ return a / length(a); }
__device__ inline float3 reflect(const float3 &I, const float3 &N){ return I - N * 2.0f * dot(N, I); }
__device__ inline void swap(float &a, float &b){ float temp = a; a = b; b = temp; }

__device__ inline float3 refract(const float3 &I, const float3 &N, float eta){
    float dotNI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
    if(k < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    return I * eta - N * (eta * dotNI + sqrtf(k));
}

__device__ inline float3 to_f3(const CudaVec3 &v){ return make_float3(v.x, v.y, v.z); }
__device__ inline CudaVec3 to_CudaVec3(const float3 &v){ CudaVec3 cv; cv.x = v.x; cv.y = v.y; cv.z = v.z; return cv; }

__device__ inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

/*--------------------------
Intersection Functions (Implementations)
--------------------------*/
__device__ inline bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
    float3 oc = ro - to_f3(s.center);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.r * s.r;
    float h = b * b - c;
    if(h < 0.0f) return false;
    h = sqrtf(h);
    float t_hit = -b - h;

    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    t_hit = -b + h;
    if(t_hit > EPSILON && t_hit < max_dist){
        t = t_hit;
        return true;
    }
    return false;
}

__device__ inline bool intersect_triangle(const float3 &ro, const float3 &rd, const CudaTriangle &tri, float &t, float max_dist){
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

__device__ inline float3 check_visibility(
    float3 p1, float3 p2,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    float3 diff = p2 - p1;
    float dist = length(diff);
    float3 dir = diff / dist;

    float3 transmission = make_float3(1.0f, 1.0f, 1.0f);
    float max_d = dist - 1e-3f;
    float min_d = 1e-3f;
    float t;

    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(p1, dir, triangles[i], t, max_d)){
            if(t > min_d){
                if(triangles[i].mtl.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
                transmission = transmission * to_f3(triangles[i].mtl.Ks);
            }
        }
    }

    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(p1, dir, spheres[i], t, max_d)){
            if(t > min_d){
                if(spheres[i].mtl.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
                transmission = transmission * to_f3(spheres[i].mtl.Ks);
            }
        }
    }
    return transmission;
}

__device__ inline CudaHit find_closest_hit(
    float3 ray_point, float3 ray_dir,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt
){
    CudaHit best;
    best.hit = false;
    best.t = 1e20f;

    float t;
    float max_dist = 1e20f;

    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(ray_point, ray_dir, spheres[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl = spheres[i].mtl;
                best.pos = ray_point + ray_dir * t;
                best.normal = normalize(best.pos - to_f3(spheres[i].center));
                if(dot(best.normal, ray_dir) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }

    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(ray_point, ray_dir, triangles[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl = triangles[i].mtl;
                best.pos = ray_point + ray_dir * t;
                float3 v0 = to_f3(triangles[i].v0);
                float3 v1 = to_f3(triangles[i].v1);
                float3 v2 = to_f3(triangles[i].v2);
                best.normal = normalize(cross(v1 - v0, v2 - v0));
                if(dot(best.normal, ray_dir) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }
    return best;
}

__device__ inline float3 sample_hemisphere_cosine_device(float3 N, curandState *state){
    float3 T, B;
    if(fabs(N.z) < 0.999f) T = normalize(cross(make_float3(0, 0, 1), N));
    else T = normalize(cross(make_float3(0, 1, 0), N));
    B = cross(N, T);
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return normalize(T * x + B * y + N * z);
}

__device__ inline float3 random_in_unit_sphere_device(curandState *state){
    float3 p;
    do{
        p = make_float3(curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f);
    } while(dot(p, p) >= 1.0f);
    return p;
}

std::ostream &operator<<(std::ostream &os, const CudaVec3 &vec);
std::istream &operator>>(std::istream &is, CudaVec3 &vec);