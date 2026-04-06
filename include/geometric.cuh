#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "object.h"

#define EPSILON 1e-4f
#define PI 3.14159265358979323846f


struct CudaMaterial_Old{
    float3 Kd, Kg, Ks;
    float glossy, exp, refract, reflect;
};

enum MaterialType{ MAT_DIFFUSE, MAT_METAL, MAT_GLASS, MAT_GLOSSY_PLASTIC };

struct CudaMaterial{
    float3 base_color;
    float roughness;
    float metallic;
    float ior;
    int type;
};

struct CudaSphere{
    float3 center;
    float r;
    CudaMaterial_Old mtl_old;
    CudaMaterial mtl;
    int id;
};

struct CudaTriangle{
    float3 v0, v1, v2;
    CudaMaterial_Old mtl_old;
    CudaMaterial mtl;
    int id;
};

struct CudaHit{
    bool hit;
    float t;
    float3 pos, normal;
    CudaMaterial_Old mtl_old;
    CudaMaterial mtl;
    bool is_light;
};

struct CudaHitPoint{
    float3 pos;
    float3 normal;
    float3 throughput;
    float3 accum_flux;
    CudaMaterial_Old mtl_old;
    CudaMaterial mtl;
    int pixel_idx;
    float radius2;
    float photon_count;
    bool valid;
};

struct CudaCamera{
    float3 eye, U, V, W, UL, dx, dy;
};

struct CudaRay{ float3 point, dir; };

struct CudaLight{
    float3 pos, dir, illum;
    CudaSphere light_ball;
    float cutoff;
    int is_parallel;
};

// Host Helper Declarations
float3 to_cv3(const glm::vec3 &v);
CudaMaterial_Old to_cmtl_old(const Material &m);
CudaMaterial to_cmtl(const Material &m);
float3 normalize_cuda(const float3 &v);

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


__device__ inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

/*--------------------------
Intersection Functions (Implementations)
--------------------------*/
__device__ inline bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
    float3 oc = ro - s.center;
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
    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;

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
                if(triangles[i].mtl_old.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
                transmission = transmission * triangles[i].mtl_old.Ks;
            }
        }
    }

    for(int i = 0; i < sphere_cnt; ++i){
        if(intersect_sphere(p1, dir, spheres[i], t, max_d)){
            if(t > min_d){
                if(spheres[i].mtl_old.refract <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
                transmission = transmission * spheres[i].mtl_old.Ks;
            }
        }
    }
    return transmission;
}

__device__ inline CudaHit find_closest_hit(
    float3 ray_point, float3 ray_dir,
    const CudaSphere *spheres, int sphere_cnt,
    const CudaTriangle *triangles, int tri_cnt,
    const CudaLight *lights, int light_cnt
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
                best.mtl_old = spheres[i].mtl_old;
                best.mtl = spheres[i].mtl;
                best.pos = ray_point + ray_dir * t;
                best.normal = normalize(best.pos - spheres[i].center);
                best.is_light = false;
                if(dot(best.normal, ray_dir) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }

    for(int i = 0; i < light_cnt; ++i){
        if(intersect_sphere(ray_point, ray_dir, lights[i].light_ball, t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl_old.Kd = lights[i].illum;
                best.mtl.base_color = lights[i].illum;
                best.pos = ray_point + ray_dir * t;
                best.normal = normalize(best.pos - lights[i].light_ball.center);
                best.is_light = true;
                if(dot(best.normal, ray_dir) > 0.0f) best.normal = best.normal * -1.0f;
            }
        }
    }

    for(int i = 0; i < tri_cnt; ++i){
        if(intersect_triangle(ray_point, ray_dir, triangles[i], t, max_dist)){
            if(t < best.t){
                best.hit = true;
                best.t = t;
                best.mtl_old = triangles[i].mtl_old;
                best.mtl = triangles[i].mtl;
                best.pos = ray_point + ray_dir * t;
                float3 v0 = triangles[i].v0;
                float3 v1 = triangles[i].v1;
                float3 v2 = triangles[i].v2;
                best.normal = normalize(cross(v1 - v0, v2 - v0));
                best.is_light = false;
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

// 1. Schlick Fresnel 近似
__device__ inline float3 fresnel_schlick(float cosTheta, float3 F0){
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * powf(fmaxf(0.0f, 1.0f - cosTheta), 5.0f);
}

// 2. GGX Normal Distribution Function (NDF)
__device__ inline float ndf_ggx(float NdotH, float roughness){
    float r = fmaxf(0.01f, roughness);

    float a = r * r;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);

    return nom / (PI * denom * denom + 1e-12f);
}

// 3. GGX Geometry (Schlick-GGX approximation)
__device__ inline float geometry_schlick_ggx(float NdotV, float roughness){
    // 針對 Path Tracing 的 k 值調整
    float a = roughness;
    float k = (a * a) / 2.0f;
    float nom = NdotV;
    float denom = NdotV * (1.0f - k) + k;
    return nom / denom;
}

__device__ inline float geometry_smith(float NdotV, float NdotL, float roughness){
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

__device__ inline float3 bsdf_evaluate(const CudaMaterial &mtl, float3 wo, float3 wi, float3 N){
    if(mtl.ior > 0.0f) return make_float3(0.0f, 0.0f, 0.0f); // 完美折射/反射不具有可評估的連續機率

    float NdotL = fmaxf(0.0f, dot(N, wi));
    float NdotV = fmaxf(0.0f, dot(N, wo));
    if(NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 H = normalize(wo + wi);
    float NdotH = fmaxf(0.0f, dot(N, H));
    float VdotH = fmaxf(0.0f, dot(wo, H));

    // 基礎反射率 F0 (非金屬通常為 0.04，金屬則為本身的基礎色)
    float metallic = mtl.metallic; // 假設你已在 CudaMaterial 加入 metallic 與 roughness
    float3 F0 = make_float3(0.04f, 0.04f, 0.04f);
    F0 = F0 * (1.0f - metallic) + mtl.base_color * metallic;

    // GGX Specular
    float D = ndf_ggx(NdotH, mtl.roughness);
    float G = geometry_smith(NdotV, NdotL, mtl.roughness);
    float3 F = fresnel_schlick(VdotH, F0);

    float3 numerator = F * D * G;
    float denominator = 4.0f * NdotV * NdotL + 1e-7f;
    float3 specular = numerator / denominator;

    // Lambertian Diffuse (能量守恆：扣除被反射的 F)
    float3 kD = make_float3(1.0f, 1.0f, 1.0f) - F;
    kD = kD * (1.0f - metallic); // 金屬內部無漫反射
    float3 diffuse = kD * mtl.base_color / PI;

    return diffuse + specular;
}

__device__ inline float bsdf_pdf(const CudaMaterial &mtl, float3 wo, float3 wi, float3 N){
    if(mtl.ior > 0.0f) return 0.0f;

    float NdotL = dot(N, wi);
    float NdotV = dot(N, wo);
    if(NdotL <= 0.0f || NdotV <= 0.0f) return 0.0f;

    float3 H = normalize(wo + wi);
    float NdotH = fmaxf(0.0f, dot(N, H));
    float VdotH = fmaxf(0.0f, dot(wo, H));

    float pdf_diffuse = NdotL / PI;
    float D = ndf_ggx(NdotH, mtl.roughness);
    float pdf_specular = (D * NdotH) / (4.0f * VdotH + 1e-7f);

    // [關鍵修正] 改用固定的 0.5 權重，確保兩種 Lobe 都有機會被採樣到
    return 0.5f * pdf_diffuse + 0.5f * pdf_specular;
}

__device__  inline void bsdf_sample(
    const CudaMaterial &mtl, float3 wo, float3 N, curandState *state, float current_ior,
    float3 &wi, float3 &bsdf_val, float &pdf, bool &is_delta, float &new_ior
){
    is_delta = false;
    new_ior = current_ior; 

    if(mtl.ior > 0.0f){
        is_delta = true;
        pdf = 1.0f;

        float3 I = wo * -1.0f;
        float3 hit_normal = N;
        float n1 = current_ior;
        float n2 = mtl.ior;

        float cosNI = dot(I, hit_normal);
        if(cosNI > 0.0f){
            swap(n1, n2);
            hit_normal = hit_normal * -1.0f;
            cosNI = dot(I, hit_normal);
        }

        float eta = n1 / n2;
        float3 refracted_dir = refract(I, hit_normal, eta);

        if(length(refracted_dir) > 0.0f){
            wi = refracted_dir;
            new_ior = mtl.ior; // 更新折射率
        }
        else{
            wi = reflect(I, hit_normal); // 全反射 (TIR)
        }

        bsdf_val = make_float3(1.0f, 1.0f, 1.0f);
        return;
    }



    float r = curand_uniform(state);
    
    r = curand_uniform(state);
    if(r <= mtl.metallic){
        is_delta = true;
        pdf = 1.0f;
        wi = reflect(wo * -1.0f, N);
        bsdf_val = mtl.base_color;
        return;
    }
    
    float prob_diffuse = 0.5f;
    r = curand_uniform(state);
    if(r < prob_diffuse){
        wi = sample_hemisphere_cosine_device(N, state);
    }
    else{

        float r1 = curand_uniform(state);
        float r2 = curand_uniform(state);

        float r3 = fmaxf(0.01f, mtl.roughness);
        float alpha = r3 * r3;

        float phi = 2.0f * PI * r1;
        float cosTheta = sqrtf(fmaxf(0.0f, (1.0f - r2) / (1.0f + (alpha * alpha - 1.0f) * r2)));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

        float3 H_local = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

        float3 T, B;
        if(fabs(N.z) < 0.999f) T = normalize(cross(make_float3(0, 0, 1), N));
        else T = normalize(cross(make_float3(0, 1, 0), N));
        B = cross(N, T);

        float3 H = normalize(T * H_local.x + B * H_local.y + N * H_local.z);
        wi = normalize(reflect(wo * -1.0f, H));

        if(dot(wi, N) <= 0.0f){
            pdf = 0.0f;
            bsdf_val = make_float3(0.0f, 0.0f, 0.0f);
            return;
        }
    }

    pdf = bsdf_pdf(mtl, wo, wi, N);
    bsdf_val = bsdf_evaluate(mtl, wo, wi, N);
}

std::ostream &operator<<(std::ostream &os, const float3 &dir);
std::istream &operator>>(std::istream &is, float3 &dir);