#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "object.h"

#define EPSILON 1e-4f
#define PI 3.14159265358979323846f

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct CudaMaterial_Old{
    float3 Kd, Kg, Ks;
    float glossy, exp, refract, reflect;
};

enum MaterialType{ MAT_DIFFUSE, MAT_DIELECTRIC, MAT_CONDUCTOR, MAT_UBER };
struct CudaMaterial{
    float3 base_color; // Albedo / F0
    float roughness;   // Perceptual roughness (0.0 ~ 1.0)
    float metallic;    // 0.0 為絕緣體，1.0 為導體(金屬)
    float eta;         // 折射率 (IOR)，空氣約為 1.0，玻璃約為 1.5
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
    float3 wo;
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
CudaMaterial_Old to_cmtl_old(const Material_Old &m);
CudaMaterial to_cmtl(const Material &m);
float3 normalize_cuda(const float3 &v);

// =========================================================
// Device Inline Implementations (必須放在 Header)
// =========================================================

HOST_DEVICE inline float3 operator+(const float3 &a, const float3 &b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
HOST_DEVICE inline float3 operator-(const float3 &a, const float3 &b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
HOST_DEVICE inline float3 operator*(const float3 &a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
HOST_DEVICE inline float3 operator*(const float3 &a, const float3 &b){ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
HOST_DEVICE inline float3 operator/(const float3 &a, float b){ return make_float3(a.x / b, a.y / b, a.z / b); }
HOST_DEVICE inline float dot(const float3 &a, const float3 &b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
HOST_DEVICE inline float3 cross(const float3 &a, const float3 &b){ return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HOST_DEVICE inline float length(const float3 &a){ return sqrtf(dot(a, a)); }
HOST_DEVICE inline float3 normalize(const float3 &a){ return a / length(a); }
HOST_DEVICE inline float3 reflect(const float3 &I, const float3 &N){ return I - N * 2.0f * dot(N, I); }
HOST_DEVICE inline void swap(float &a, float &b){ float temp = a; a = b; b = temp; }

HOST_DEVICE inline float3 refract(const float3 &I, const float3 &N, float eta){
    float dotNI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
    if(k < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    return I * eta - N * (eta * dotNI + sqrtf(k));
}


HOST_DEVICE inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

// =========================================================
// pbrt-v4 Local Space Core Math
// =========================================================

// 局部座標系轉換輔助
HOST_DEVICE inline void build_local_frame(float3 N, float3 &T, float3 &B){
    if(fabs(N.z) < 0.999f) T = normalize(cross(make_float3(0, 0, 1), N));
    else T = normalize(cross(make_float3(0, 1, 0), N));
    B = cross(N, T);
}
HOST_DEVICE inline float3 world_to_local(float3 v, float3 T, float3 B, float3 N){
    return make_float3(dot(v, T), dot(v, B), dot(v, N));
}
HOST_DEVICE inline float3 local_to_world(float3 v, float3 T, float3 B, float3 N){
    return make_float3(
        T.x * v.x + B.x * v.y + N.x * v.z,
        T.y * v.x + B.y * v.y + N.y * v.z,
        T.z * v.x + B.z * v.y + N.z * v.z
    );
}

// 局部空間的三角函數 (pbrt-v4 style)
HOST_DEVICE inline float CosTheta(const float3 &w){ return w.z; }
HOST_DEVICE inline float Cos2Theta(const float3 &w){ return w.z * w.z; }
HOST_DEVICE inline float AbsCosTheta(const float3 &w){ return fabs(w.z); }
HOST_DEVICE inline float Sin2Theta(const float3 &w){ return fmaxf(0.0f, 1.0f - Cos2Theta(w)); }
HOST_DEVICE inline float SinTheta(const float3 &w){ return sqrtf(Sin2Theta(w)); }
HOST_DEVICE inline float TanTheta(const float3 &w){ return SinTheta(w) / (CosTheta(w) + 1e-7f); }
HOST_DEVICE inline float Tan2Theta(const float3 &w){ return Sin2Theta(w) / (Cos2Theta(w) + 1e-7f); }

// 精確的絕緣體 Fresnel (FrDielectric)
HOST_DEVICE inline float FrDielectric(float cosThetaI, float etaI, float etaT){
    cosThetaI = fmaxf(-1.0f, fminf(1.0f, cosThetaI));
    bool entering = cosThetaI > 0.0f;
    if(!entering){
        swap(etaI, etaT);
        cosThetaI = fabs(cosThetaI);
    }
    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if(sinThetaT >= 1.0f) return 1.0f; // 全反射 (TIR)
    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

// 金屬的 Schlick 近似 (pbrt 也允許使用這個來簡化複雜 IOR)
HOST_DEVICE inline float3 FrSchlick(float cosThetaI, float3 R0){
    float c = fmaxf(0.0f, 1.0f - cosThetaI);
    float c5 = c * c * c * c * c;
    return R0 + (make_float3(1.0f, 1.0f, 1.0f) - R0) * c5;
}


// =========================================================
// pbrt-v4 Trowbridge-Reitz (GGX) Microfacet Model
// =========================================================
HOST_DEVICE inline float RoughnessToAlpha(float roughness){
    float x = fmaxf(roughness, 1e-3f);
    return x * x;
}

HOST_DEVICE inline float TrowbridgeReitzD(float3 wh, float alpha){
    float tan2Theta = Tan2Theta(wh);
    if(isinf(tan2Theta)) return 0.0f;
    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    float e = (cos4Theta * (alpha * alpha + tan2Theta * tan2Theta));
    if(e < 1e-12f) return 0.0f;
    return (alpha * alpha) / (PI * e);
}

HOST_DEVICE inline float TrowbridgeReitzLambda(float3 w, float alpha){
    float absTanTheta = fabs(TanTheta(w));
    if(isinf(absTanTheta)) return 0.0f;
    float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
}

HOST_DEVICE inline float TrowbridgeReitzG(float3 wo, float3 wi, float alpha){
    return 1.0f / (1.0f + TrowbridgeReitzLambda(wo, alpha) +
        TrowbridgeReitzLambda(wi, alpha));
}

// 可見法線採樣 (VNDF) - 採樣可見的微平面法線
HOST_DEVICE inline float3 SampleTrowbridgeReitzVisibleNormal(float3 wo, float alpha, float u1, float u2){
    // 拉伸視角
    float3 V = normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));

    // 建立正交基底
    float3 T1 = (V.z < 0.9999f) ? normalize(cross(make_float3(0, 0, 1), V)) : make_float3(1, 0, 0);
    float3 T2 = cross(V, T1);

    // 圓盤採樣
    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + V.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // 投影回半球
    float3 Nh = T1 * t1 + T2 * t2 + V * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));

    // 取消拉伸
    return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}

HOST_DEVICE inline bool is_valid_color(float3 c){
    return !(isnan(c.x) || isnan(c.y) || isnan(c.z) ||
        isinf(c.x) || isinf(c.y) || isinf(c.z) ||
        c.x < 0.0f || c.y < 0.0f || c.z < 0.0f);
}

HOST_DEVICE inline float3 clamp_radiance(float3 c, float max_val){
    float max_channel = fmaxf(c.x, fmaxf(c.y, c.z));
    if(max_channel > max_val){
        return c * (max_val / max_channel);
    }
    return c;
}

/*--------------------------
Intersection Functions (Implementations)
--------------------------*/
HOST_DEVICE inline bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
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

HOST_DEVICE inline bool intersect_triangle(const float3 &ro, const float3 &rd, const CudaTriangle &tri, float &t, float max_dist){
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

HOST_DEVICE inline float3 check_visibility(
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

HOST_DEVICE inline CudaHit find_closest_hit(
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

// 移除 curandState，改為傳入 u1, u2
HOST_DEVICE inline float3 sample_hemisphere_cosine_device(float3 N, float u1, float u2){
    float3 T, B;
    if(fabs(N.z) < 0.999f) T = normalize(cross(make_float3(0, 0, 1), N));
    else T = normalize(cross(make_float3(0, 1, 0), N));
    B = cross(N, T);

    // 直接使用傳入的隨機數
    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return normalize(T * x + B * y + N * z);
}

#ifdef __CUDACC__
__device__ inline float3 random_in_unit_sphere_device(curandState *state){
    float3 p;
    do{
        p = make_float3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
    } while(dot(p, p) >= 1.0f);
    return p;
}
#endif

// =========================================================
// pbrt-v4 Style BSDF API
// =========================================================
HOST_DEVICE inline float3 bsdf_evaluate(const CudaMaterial &mtl, float3 wo_w, float3 wi_w, float3 N){
    // 建立局部空間
    float3 T, B;
    build_local_frame(N, T, B);
    float3 wo = world_to_local(wo_w, T, B, N);
    float3 wi = world_to_local(wi_w, T, B, N);

    if(CosTheta(wo) == 0.0f || CosTheta(wi) == 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    if(mtl.eta > 0.0f && mtl.roughness < 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    float alpha = RoughnessToAlpha(mtl.roughness);

    float3 wh_vec = wo + wi;
    if(length(wh_vec) < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
    float3 wh = normalize(wh_vec);
    if(wh.z < 0.0f) wh = wh * -1.0f;

    float3 diffuse = mtl.base_color / PI * (1.0f - mtl.metallic);
    if(wo.z * wi.z < 0.0f) diffuse = make_float3(0.0f, 0.0f, 0.0f);

    float D = TrowbridgeReitzD(wh, alpha);
    float G = TrowbridgeReitzG(wo, wi, alpha);

    float3 F;
    if(mtl.metallic > 0.0f){
        F = FrSchlick(AbsCosTheta(wo), mtl.base_color);
    }
    else{
        float fr = FrDielectric(dot(wo, wh), 1.0f, mtl.eta);
        F = make_float3(fr, fr, fr);
    }

    float3 specular = (F * D * G) / fmaxf(4.0f * AbsCosTheta(wo) * AbsCosTheta(wi), 1e-4f);

    if(wo.z * wi.z > 0.0f) return diffuse + specular;
    return diffuse;
}

HOST_DEVICE inline float bsdf_pdf(const CudaMaterial &mtl, float3 wo_w, float3 wi_w, float3 N){
    float3 T, B;
    build_local_frame(N, T, B);
    float3 wo = world_to_local(wo_w, T, B, N);
    float3 wi = world_to_local(wi_w, T, B, N);

    if(CosTheta(wo) * CosTheta(wi) <= 0.0f) return 0.0f;
    if(mtl.eta > 0.0f && mtl.roughness < 0.001f) return 0.0f;

    float alpha = RoughnessToAlpha(mtl.roughness);
    float3 wh_vec = wo + wi;
    if(length(wh_vec) < 1e-6f) return 0.0f;
    float3 wh = normalize(wh_vec);
    if(wh.z < 0.0f) wh = wh * -1.0f;
    float pdf_diffuse = AbsCosTheta(wi) / PI;

    // VNDF 的 PDF 計算: D * G1 * max(0, dot(wo, wh)) / cos(wo)
    // 轉換為 wi 的 PDF: pdf_wh / (4 * dot(wo, wh))
    float G1 = 1.0f / (1.0f + TrowbridgeReitzLambda(wo, alpha));
    float pdf_wh = TrowbridgeReitzD(wh, alpha) * G1 * fmaxf(0.0f, dot(wo, wh)) / AbsCosTheta(wo);
    float pdf_specular = pdf_wh / (4.0f * dot(wo, wh) + 1e-7f);

    float spec_weight = mtl.metallic > 0.0f ? 1.0f : 0.5f;
    float diff_weight = 1.0f - spec_weight;

    return diff_weight * pdf_diffuse + spec_weight * pdf_specular;
}

HOST_DEVICE inline void bsdf_sample(
    const CudaMaterial &mtl, float3 wo_w, float3 N,
    float u_rr, float u1, float u2, float current_eta, // [修改] 替換 state
    float3 &wi_w, float3 &bsdf_val, float &pdf, bool &is_delta, float &new_eta
){
    is_delta = false;
    new_eta = current_eta;

    // 建立局部空間
    float3 T, B;
    build_local_frame(N, T, B);
    float3 wo = world_to_local(wo_w, T, B, N);
    float3 wi;

    if(mtl.eta > 0.0f && mtl.roughness < 0.001f && mtl.metallic < 0.01f){
        is_delta = true;
        float F = FrDielectric(CosTheta(wo), current_eta, mtl.eta);

        if(u_rr < F){
            // 反射
            wi = make_float3(-wo.x, -wo.y, wo.z);
            pdf = F;
            bsdf_val = make_float3(F, F, F) / AbsCosTheta(wi);
        }
        else{
            float eta = CosTheta(wo) > 0.0f ? (current_eta / mtl.eta) : (mtl.eta / current_eta);
            float sin2ThetaI = fmaxf(0.0f, 1.0f - Cos2Theta(wo));
            float sin2ThetaT = eta * eta * sin2ThetaI;
            if(sin2ThetaT >= 1.0f){ pdf = 0.0f; return; } // 全反射處理

            float cosThetaT = sqrtf(1.0f - sin2ThetaT);
            if(CosTheta(wo) > 0.0f) cosThetaT = -cosThetaT;

            wi = make_float3(-eta * wo.x, -eta * wo.y, cosThetaT);
            new_eta = (CosTheta(wo) > 0.0f) ? mtl.eta : 1.0f; // 假設外部為空氣(1.0)

            pdf = 1.0f - F;
            bsdf_val = mtl.base_color * (1.0f - F) / AbsCosTheta(wi);
        }

        wi_w = local_to_world(wi, T, B, N);
        return;
    }

    if(mtl.metallic > 0.99f && mtl.roughness < 0.001f){
        is_delta = true;
        wi = make_float3(-wo.x, -wo.y, wo.z);
        pdf = 1.0f;
        bsdf_val = FrSchlick(AbsCosTheta(wo), mtl.base_color) / AbsCosTheta(wi);
        wi_w = local_to_world(wi, T, B, N);
        return;
    }

    float alpha = RoughnessToAlpha(mtl.roughness);
    float spec_weight = mtl.metallic > 0.0f ? 1.0f : 0.5f;

    if(u_rr < spec_weight){
        // Specular (VNDF 採樣)

        float3 wh = SampleTrowbridgeReitzVisibleNormal(wo.z > 0 ? wo : wo * -1.0f, alpha, u1, u2);
        if(wo.z < 0.0f) wh = wh * -1.0f;

        wi = reflect(wo * -1.0f, wh);
        if(wo.z * wi.z <= 0.0f){ pdf = 0.0f; return; }
    }
    else{
        // Diffuse (Cosine-weighted)
        float r = sqrtf(u1);
        float phi = 2.0f * PI * u2;
        wi = make_float3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
        if(wo.z < 0.0f) wi.z *= -1.0f;
    }

    wi_w = local_to_world(wi, T, B, N);
    pdf = bsdf_pdf(mtl, wo_w, wi_w, N);
    bsdf_val = bsdf_evaluate(mtl, wo_w, wi_w, N);
}

std::ostream &operator<<(std::ostream &os, const float3 &dir);
std::istream &operator>>(std::istream &is, float3 &dir);