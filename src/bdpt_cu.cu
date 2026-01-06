#include "bdpt_cu.cuh"
#include <curand_kernel.h>
#include <cstdio>
#define EPSILON 1e-4f
#define PI 3.14159265358979323846f
#define BLOCK_SIZE 256

/*--------------------------
Vector Math Functions
--------------------------*/
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

    if(k < 0.0f){
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    return I * eta - N * (eta * dotNI + sqrtf(k));
}
__device__ inline float3 to_f3(const CudaVec3 &v){ return make_float3(v.x, v.y, v.z); }
__device__ inline CudaVec3 to_CudaVec3(const float3 &v){ CudaVec3 cv; cv.x = v.x; cv.y = v.y; cv.z = v.z; return cv; }
__device__ inline float3 pow_f3(const float3 &base, float exp){
    return make_float3(powf(base.x, exp), powf(base.y, exp), powf(base.z, exp));
}

/*--------------------------
RNG Functions
--------------------------*/
__global__ void init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*--------------------------
Intersection Functions
--------------------------*/
__device__ bool intersect_sphere(const float3 &ro, const float3 &rd, const CudaSphere &s, float &t, float max_dist){
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

__device__ bool intersect_triangle(const float3 &ro, const float3 &rd, const CudaTriangle &tri, float &t, float max_dist){
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

__device__ float3 check_visibility(
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
                // Simple attenuation for glass
                transmission = transmission * to_f3(spheres[i].mtl.Ks);
            }
        }
    }
    return transmission;
}

__device__ CudaHit find_closest_hit(
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

__device__ float3 sample_hemisphere_cosine_device(float3 N, curandState *state){
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

__device__ float3 random_in_unit_sphere_device(curandState *state){
    float3 p;
    do{
        p = make_float3(curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f,
            curand_uniform(state) * 2.0f - 1.0f);
    } while(dot(p, p) >= 1.0f);
    return p;
}


/*--------------------------
Light Tracing Kernel
--------------------------*/
__global__ void cuda_light_trace(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    const CudaVec3 min_bound, const CudaVec3 max_bound,
    CudaLightVertex *cuda_light_vertices,
    curandState *states,
    int max_depth, int total_paths
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_paths) return;

    int light_idx = idx % num_lights;
    CudaLight light = cuda_lights[light_idx];


    int path_base_idx = idx * max_depth;

    float3 ray_point, ray_dir;
    float ray_refract = 1.0f;
    curandState localState = states[idx];

    if(light.is_parallel){
        ray_dir = normalize(to_f3(light.dir));

        float3 scene_center = (to_f3(min_bound) + to_f3(max_bound)) * 0.5f;
        float3 diag = to_f3(max_bound) - to_f3(min_bound);
        float scene_radius = length(diag) * 0.5f;

        float3 w = ray_dir;
        float3 u, v;

        if(fabs(w.x) > 0.9f) u = make_float3(0.0f, 1.0f, 0.0f);
        else u = make_float3(1.0f, 0.0f, 0.0f);

        v = normalize(cross(w, u));
        u = normalize(cross(v, w));

        float r1 = curand_uniform(&localState);
        float r2 = curand_uniform(&localState);

        float plane_size = scene_radius * 2.0f;
        float offset_u = (r1 - 0.5f) * plane_size;
        float offset_v = (r2 - 0.5f) * plane_size;

        ray_point = scene_center - ray_dir * (scene_radius * 2.0f) + u * offset_u + v * offset_v;
    }
    else{
        ray_point = to_f3(light.pos);

        float3 w = normalize(to_f3(light.dir));
        float3 u, v;

        if(fabs(w.x) > 0.9f) u = make_float3(0.0f, 1.0f, 0.0f);
        else u = make_float3(1.0f, 0.0f, 0.0f);

        v = normalize(cross(w, u));
        u = normalize(cross(v, w));

        float u1 = curand_uniform(&localState);
        float u2 = curand_uniform(&localState);
        float theta = acosf(1.0f - u1 * (1.0f - cosf(light.cutoff)));
        float phi = 2.0f * PI * u2;

        float3 local_dir = make_float3(
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        );

        ray_dir = normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
    }

    float3 throughput = to_f3(light.illum);

    CudaLightVertex &vertex0 = cuda_light_vertices[path_base_idx];
    vertex0.pos = to_CudaVec3(ray_point);
    vertex0.normal = to_CudaVec3(ray_dir);
    vertex0.throughput = to_CudaVec3(throughput);
    vertex0.is_light_source = true;
    vertex0.source_cutoff = light.cutoff;
    vertex0.is_parallel = light.is_parallel;

    float3 last_normal = ray_dir;
    float3 last_pos = ray_point;
    float last_pdf_omega = 1.0f / PI;


    for(int depth = 1; depth < max_depth; depth++){
        CudaLightVertex &vertex = cuda_light_vertices[path_base_idx + depth];

        CudaHit hit = find_closest_hit(ray_point, ray_dir, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
        if(!hit.hit) break;
        if(length(throughput) < 1e-4f) break;

        float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
        float dist = sqrtf(dist2);
        if(dist2 < 1e-6f) break;

        float cos_at_hit = abs(dot(hit.normal, ray_dir * -1.0f));
        float cos_at_prev = abs(dot(last_normal, ray_dir));

        float pdf_fwd = last_pdf_omega * cos_at_hit / dist2;
        float pdf_rev_omega = cos_at_hit / PI;
        float pdf_rev = pdf_rev_omega * cos_at_prev / dist2;

        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
            continue;
        }
        if(hit.mtl.refract > 0.0f){
            float3 refracted_dir;
            float3 I = ray_dir, N = hit.normal;
            float n1 = ray_refract;
            float n2 = hit.mtl.refract;

            float cosNI = dot(I, N);
            if(cosNI > 0.0f){
                swap(n1, n2);
                N = N * -1.0f;
                cosNI = dot(I, N);
            }
            float eta = n1 / n2;
            refracted_dir = refract(I, N, eta);
            if(length(refracted_dir) > 0.0f){
                ray_point = hit.pos - hit.normal * EPSILON;
                ray_dir = refracted_dir;
                ray_refract = hit.mtl.refract;
            }
            else{
                ray_point = hit.pos + hit.normal * EPSILON;
                ray_dir = reflect(ray_dir, hit.normal);
            }
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl.glossy){
            float3 perfect_reflect = reflect(ray_dir, hit.normal);
            float roughness = (hit.mtl.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl.exp * 0.005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            ray_dir = normalize(perfect_reflect + jitter);
            if(dot(ray_dir, hit.normal) < 0.0f){
                ray_dir = ray_dir - hit.normal * dot(ray_dir, hit.normal) * 2.0f;
                ray_dir = normalize(ray_dir);
            }
            ray_point = hit.pos + ray_dir * EPSILON;
        }
        else{
            ray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            ray_point = hit.pos + ray_dir * EPSILON;
            throughput = throughput * to_f3(hit.mtl.Kd); // Diffuse attenuation

            vertex.pos = to_CudaVec3(hit.pos);
            vertex.normal = to_CudaVec3(hit.normal);
            vertex.throughput = to_CudaVec3(throughput);
            vertex.mtl = hit.mtl;
            vertex.is_light_source = false;
            vertex.pdf_fwd = pdf_fwd;
            vertex.pdf_rev = pdf_rev;

            last_pdf_omega = pdf_rev_omega;
            last_normal = hit.normal;
            last_pos = hit.pos;
        }
    }
    states[idx] = localState;
}


/*--------------------------
Eye Trace Kernel (Fixed Logic)
--------------------------*/
__global__ void cuda_eye_trace_and_connect(
    const CudaLight *cuda_lights, int num_lights,
    const CudaSphere *cuda_spheres, int num_spheres,
    const CudaTriangle *cuda_triangles, int num_triangles,
    const CudaVec3 min_bound, const CudaVec3 max_bound,
    const CudaCamera cuda_camera,
    CudaLightVertex *cuda_light_vertices, int num_light_vertices,
    CudaEyeVertex *cuda_eye_vertices,
    curandState *states,
    int W, int H,
    int max_depth, CudaVec3 *cuda_image
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    // Generate Eye Ray
    int px = idx % W;
    int py = idx / W;

    int path_base_idx = idx * max_depth;
    curandState localState = states[idx];

    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);

    float3 ray_point = to_f3(cuda_camera.eye);

    float3 pixel_pos = to_f3(cuda_camera.UL) +
        to_f3(cuda_camera.dx) * pixel_x +
        to_f3(cuda_camera.dy) * pixel_y;

    float3 ray_dir = normalize(pixel_pos - to_f3(cuda_camera.eye));
    float ray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    CudaVec3 *pixel = &cuda_image[idx];

    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

    // Connect and Trace
    for(int depth = 0; depth < max_depth; depth++){
        CudaEyeVertex &vertex = cuda_eye_vertices[path_base_idx + depth];

        CudaHit hit = find_closest_hit(ray_point, ray_dir, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
        if(!hit.hit) break;

        vertex.pos = to_CudaVec3(hit.pos);
        vertex.normal = to_CudaVec3(hit.normal);
        vertex.throughput = to_CudaVec3(throughput);
        vertex.mtl = hit.mtl;
        vertex.pdf_fwd = 1.0f; // Placeholder
        vertex.pdf_rev = 1.0f; // Placeholder

        // Connect to Light Vertices
        float3 total_L = make_float3(0.0f, 0.0f, 0.0f);
        for(int light_idx = 0; light_idx < num_light_vertices; light_idx++){
            CudaLightVertex lv = cuda_light_vertices[light_idx];
            float3 lv_pos = to_f3(lv.pos);
            float3 lv_normal = to_f3(lv.normal);
            float3 lv_throughput = to_f3(lv.throughput);

            CudaEyeVertex &ev = vertex;
            float3 ev_pos = to_f3(ev.pos);
            float3 ev_normal = to_f3(ev.normal);
            float3 ev_throughput = to_f3(ev.throughput);
            float3 ev_kd = to_f3(ev.mtl.Kd);

            float3 fE = ev_kd / PI; // Diffuse BRDF

            float3 d_vec = lv_pos - ev_pos;
            float dist2 = dot(d_vec, d_vec);
            if(dist2 < 1e-6f) continue;

            float dist = sqrtf(dist2);
            float3 wi = d_vec / dist;

            float cosE = fmaxf(0.0f, dot(ev_normal, wi));
            float cosL = fmaxf(0.0f, dot(lv_normal, wi * -1.0f));

            if(cosE <= 0.0f || cosL <= 0.0f) continue;
            if(lv.is_light_source && lv.source_cutoff > 0.0f && !lv.is_parallel){
                float3 light_dir = normalize(to_f3(cuda_lights[light_idx % num_lights].dir));
                float cos_theta = dot(light_dir, wi * -1.0f);
                float cutoff_cos = cosf(lv.source_cutoff);
                if(cos_theta < cutoff_cos) continue;
            }
            float3 transmittance = check_visibility(ev_pos + ev_normal * EPSILON, lv_pos + lv_normal * EPSILON, cuda_spheres, num_spheres, cuda_triangles, num_triangles);
            if(transmittance.x <= 0.0f && transmittance.y <= 0.0f && transmittance.z <= 0.0f) continue;
            float G = (cosE * cosL) / dist2;

            float3 contrib = ev_throughput * fE * G * lv_throughput * transmittance;
            total_L = total_L + contrib;
        }
        final_color = final_color + total_L;

        // Update Ray for next bounce
        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
            continue;
        }
        if(hit.mtl.refract > 0.0f){
            float3 refracted_dir;
            float3 I = ray_dir, N = hit.normal;
            float n1 = ray_refract;
            float n2 = hit.mtl.refract;

            float cosNI = dot(I, N);
            if(cosNI > 0.0f){
                swap(n1, n2);
                N = N * -1.0f;
                cosNI = dot(I, N);
            }
            float eta = n1 / n2;
            refracted_dir = refract(I, N, eta);
            if(length(refracted_dir) > 0.0f){
                ray_point = hit.pos - hit.normal * EPSILON;
                ray_dir = refracted_dir;
                ray_refract = hit.mtl.refract;
            }
            else{
                ray_point = hit.pos + hit.normal * EPSILON;
                ray_dir = reflect(ray_dir, hit.normal);
            }
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl.glossy){
            float3 perfect_reflect = reflect(ray_dir, hit.normal);
            float roughness = (hit.mtl.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl.exp * 0.005f + .001f);
            float3 jitter = random_in_unit_sphere_device(&localState) * roughness;
            ray_dir = normalize(perfect_reflect + jitter);
            if(dot(ray_dir, hit.normal) < 0.0f){
                ray_dir = ray_dir - hit.normal * dot(ray_dir, hit.normal) * 2.0f;
                ray_dir = normalize(ray_dir);
            }
            ray_point = hit.pos + ray_dir * EPSILON;
        }
        else{
            ray_dir = sample_hemisphere_cosine_device(hit.normal, &localState);
            ray_point = hit.pos + ray_dir * EPSILON;
            throughput = throughput * to_f3(hit.mtl.Kd); // Diffuse attenuation
        }

    }

    *pixel = to_CudaVec3(final_color);
    states[idx] = localState;
}

void bdpt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    CudaVec3 scene_min, CudaVec3 scene_max,
    const CudaCamera cuda_camera, CudaVec3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth
){
    // ---------------------------------------------------------
    // 1. Setup Scene Data (保持不變)
    // ---------------------------------------------------------
    CudaLight *d_lights = nullptr;
    CudaSphere *d_spheres = nullptr;
    CudaTriangle *d_triangles = nullptr;

    if(num_lights > 0){
        cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
        cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
    }
    if(num_spheres > 0){
        cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
        cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    }
    if(num_triangles > 0){
        cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
        cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);
    }

    // ---------------------------------------------------------
    // 2. Light Tracing Phase
    // ---------------------------------------------------------
    // 計算總 Light Paths 數量 (光源數 * 採樣數)
    int total_light_paths = num_lights * light_sample;

    // Light Vertex Buffer 大小
    size_t total_light_vertices_size = (size_t) total_light_paths * light_depth;

    CudaLightVertex *d_cuda_light_vertices;
    cudaError_t err = cudaMalloc(&d_cuda_light_vertices, sizeof(CudaLightVertex) * total_light_vertices_size);
    if(err != cudaSuccess){ printf("Malloc LightVertices failed: %s\n", cudaGetErrorString(err)); return; }
    cudaMemset(d_cuda_light_vertices, 0, sizeof(CudaLightVertex) * total_light_vertices_size);

    // 準備 RNG States
    // 注意：這裡我們分配 "最大需求" 的空間，通常是 W*H (因為像素數量遠大於光源路徑數)
    // 這樣可以讓 Light Trace 和 Eye Trace 共用同一個 state buffer
    size_t max_threads = (size_t) W * H;
    if(total_light_paths > max_threads) max_threads = total_light_paths;

    curandState *d_states;
    cudaMalloc(&d_states, sizeof(curandState) * max_threads);

    int threads = BLOCK_SIZE;

    // --- Light Trace Launch Configuration ---
    // 使用 total_light_paths 來計算 Block 數量
    int light_blocks = (total_light_paths + threads - 1) / threads;

    // init rng for Light Trace
    init_rng << <light_blocks, threads >> > (d_states, time(NULL) + clock(), total_light_paths);    cudaDeviceSynchronize();

    cuda_light_trace << <light_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max,
        d_cuda_light_vertices, d_states,
        light_depth,
        total_light_paths // 傳入正確的路徑總數
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess) printf("Light Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 3. Eye Tracing Phase
    // ---------------------------------------------------------
    CudaEyeVertex *d_cuda_eye_vertices;
    size_t total_eye_vertices_size = (size_t) W * H * eye_depth;
    err = cudaMalloc(&d_cuda_eye_vertices, sizeof(CudaEyeVertex) * total_eye_vertices_size);
    if(err != cudaSuccess){ printf("Malloc EyeVertices failed: %s\n", cudaGetErrorString(err)); return; }

    CudaVec3 *d_image;
    cudaMalloc(&d_image, sizeof(CudaVec3) * W * H);

    // --- Eye Trace Launch Configuration ---
    // 關鍵修正：這裡必須使用 W * H 來計算 Block 數量
    int total_pixels = W * H;
    int eye_blocks = (total_pixels + threads - 1) / threads;

    // Re-seed for Eye Trace (using total_pixels)
    init_rng << <eye_blocks, threads >> > (d_states, time(NULL) + 9999, total_pixels);
    cudaDeviceSynchronize();

    printf("Eye Trace Launch: Pixels=%d, Blocks=%d\n", total_pixels, eye_blocks);

    cuda_eye_trace_and_connect << <eye_blocks, threads >> > (
        d_lights, num_lights,
        d_spheres, num_spheres,
        d_triangles, num_triangles,
        scene_min, scene_max, cuda_camera,
        d_cuda_light_vertices, total_light_vertices_size, // 注意：這裡你傳入的是 Buffer 大小，你的 Kernel 迴圈會跑這麼多次
        d_cuda_eye_vertices, d_states,
        W, H,
        eye_depth,
        d_image
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess) printf("Eye Trace Error: %s\n", cudaGetErrorString(err));

    // ---------------------------------------------------------
    // 4. Retrieve Data
    // ---------------------------------------------------------
    cudaMemcpy(h_image, d_image, sizeof(CudaVec3) * W * H, cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 5. Cleanup
    // ---------------------------------------------------------
    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_cuda_light_vertices);
    cudaFree(d_cuda_eye_vertices);
    cudaFree(d_states);
    cudaFree(d_image);
}