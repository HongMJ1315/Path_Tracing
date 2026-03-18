#include "pt_cu.cuh"
#include <cstdio>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-4f

// 初始化隨機數生成器
__global__ void pt_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}



/*--------------------------
傳統 Path Tracing Kernel (含 NEE)
--------------------------*/
__global__ void cuda_path_trace_kernel(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaCamera cam, curandState *states,
    int W, int H, int max_depth, float3 *d_image
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];


    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    // Ray Generation
    float pixel_x = (float) px + curand_uniform(&localState);
    float pixel_y = (float) py + curand_uniform(&localState);
    float3 ray_point = cam.eye;
    float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
    float3 ray_dir = normalize(pixel_pos - ray_point);
    float ray_refract = 1.0f;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    for(int depth = 0; depth < max_depth; ++depth){
        CudaHit hit = find_closest_hit(ray_point, ray_dir,
            d_spheres, num_spheres,
            d_triangles, num_triangles,
            d_lights, num_lights);
        if(!hit.hit) break;
        if(hit.is_light){
            final_color = final_color + throughput * hit.mtl.Kd;
            break;
        }
        float do_reflect = curand_uniform(&localState);
        if(hit.mtl.reflect > 0.0f && do_reflect < hit.mtl.reflect){
            ray_point = hit.pos + hit.normal * EPSILON;
            ray_dir = reflect(ray_dir, hit.normal);
            depth--;
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
            depth--;
            continue;
        }
        float do_glossy = curand_uniform(&localState);
        if(do_glossy < hit.mtl.glossy){
            float3 perfect_reflect = reflect(ray_dir, hit.normal);
            float roughness = (hit.mtl.exp > 1000.f) ? 0.0f : 1.0f / (hit.mtl.exp * 0.0005f + .001f);
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
            throughput = throughput * hit.mtl.Kd; // Diffuse attenuation
        }
    }

    d_image[idx] = final_color;
    states[idx] = localState;
}

/*--------------------------
Wrapper 實作 (保持與 BDPT 介面完全一致)
--------------------------*/
void pt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth
){
    // 1. 分配 GPU 記憶體 (與 bdpt_cu.cu 邏輯一致)
    CudaLight *d_lights;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    float3 *d_image;
    curandState *d_states;

    cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
    cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
    cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
    cudaMalloc(&d_image, sizeof(float3) * W * H);
    cudaMalloc(&d_states, sizeof(curandState) * W * H);

    cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);

    // 2. 初始化隨機數
    int total_pixels = W * H;
    int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pt_init_rng << <blocks, BLOCK_SIZE >> > (d_states, time(NULL), total_pixels);

    // 3. 執行 Path Tracing (這裡將 eye_depth 作為 PT 的 max_depth)
    cuda_path_trace_kernel << <blocks, BLOCK_SIZE >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_states, W, H, eye_depth, d_image
        );
    cudaDeviceSynchronize();

    // 4. 回傳數據
    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    // 5. 釋放資源
    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_image);
    cudaFree(d_states);
}