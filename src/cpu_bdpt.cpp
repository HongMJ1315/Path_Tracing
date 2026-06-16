#include "cpu_bdpt.h"
#include <omp.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "geometric.cuh" 
#include <chrono>

// =========================================================
// CPU 輔助函數：數值防護
// =========================================================
inline bool cpu_is_valid_color(float3 c){
    return !(std::isnan(c.x) || std::isnan(c.y) || std::isnan(c.z) ||
        std::isinf(c.x) || std::isinf(c.y) || std::isinf(c.z) ||
        c.x < 0.0f || c.y < 0.0f || c.z < 0.0f);
}

inline float3 cpu_clamp_radiance(float3 c, float max_val){
    float max_channel = std::max({ c.x, c.y, c.z });
    if(max_channel > max_val){
        float scale = max_val / max_channel;
        return make_float3(c.x * scale, c.y * scale, c.z * scale);
    }
    return c;
}

// =========================================================
// CPU 輔助函數：幾何相交與可見性測試
// =========================================================
CudaHit cpu_find_closest_hit(
    float3 ray_point, float3 ray_dir,
    std::map<int, AABB> &groups,
    const std::vector<CudaLight> &lights){
    CudaHit best;
    best.hit = false;
    best.t = 1e20f;

    Ray ray;
    ray.point = glm::vec3(ray_point.x, ray_point.y, ray_point.z);
    ray.vec = glm::vec3(ray_dir.x, ray_dir.y, ray_dir.z);

    // 1. 測試 AABB 與場景物件
    for(auto &g : groups){
        if(!g.second.intersectAABB(ray, 1e-4f, best.t)) continue;

        for(Object *obj : g.second.objs){
            float t, u, v;
            if(obj->check_intersect(ray, t, u, v, 1e-4f, best.t)){
                best.hit = true;
                best.t = t;
                best.mtl = to_cmtl(obj->mtl);

                glm::vec3 P = ray.point + ray.vec * t;
                glm::vec3 N = obj->normal_at(P, ray, u, v);

                best.pos = make_float3(P.x, P.y, P.z);
                best.normal = make_float3(N.x, N.y, N.z);
                best.is_light = false;
            }
        }
    }

    // 2. 測試光源球體
    for(const auto &light : lights){
        float t;
        if(intersect_sphere(ray_point, ray_dir, light.light_ball, t, best.t)){
            best.hit = true;
            best.t = t;
            best.mtl.base_color = light.illum;
            best.mtl.eta = 0.0f;
            best.mtl.roughness = 1.0f;
            best.mtl.metallic = 0.0f;
            best.pos = ray_point + ray_dir * t;
            best.normal = normalize(best.pos - light.light_ball.center);
            best.is_light = true;
            if(dot(best.normal, ray_dir) > 0.0f) best.normal = best.normal * -1.0f;
        }
    }
    return best;
}

float3 cpu_check_visibility(float3 p1, float3 p2, std::map<int, AABB> &groups){
    float3 diff = p2 - p1;
    float dist = length(diff);
    float3 dir = diff / dist;

    Ray ray;
    ray.point = glm::vec3(p1.x, p1.y, p1.z);
    ray.vec = glm::vec3(dir.x, dir.y, dir.z);

    float max_dist = dist - 1e-3f;
    float3 transmittance = make_float3(1.0f, 1.0f, 1.0f);

    for(auto &g : groups){
        float tMin = 1e-3f, tMax = max_dist;
        if(!g.second.intersectAABB(ray, tMin, tMax)) continue;

        for(Object *obj : g.second.objs){
            float t, u, v;
            if(obj->check_intersect(ray, t, u, v, tMin, tMax)){
                // 如果遇到不透明物體，直接阻斷光線
                if(obj->mtl.eta <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
            }
        }
    }
    return transmittance;
}

// =========================================================
// CPU 輔助函數：MIS 權重計算
// =========================================================
float cpu_calculate_mis_weight(
    const CudaEyeVertex *eye_path, int s_idx,
    const CudaLightVertex *light_path, int t_idx,
    const float3 &dir_e_to_l, float dist2,
    const float3 &camera_pos){
    if(s_idx < 0 || t_idx < 0) return 0.0f;

    const CudaEyeVertex &ev = eye_path[s_idx];
    const CudaLightVertex &lv = light_path[t_idx];

    float3 ns = normalize(ev.normal);
    float3 nt = normalize(lv.normal);

    float cos_s = fmaxf(0.0f, dot(ns, dir_e_to_l));
    float cos_t = fmaxf(0.0f, dot(nt, dir_e_to_l * -1.0f));

    if(cos_s <= 0.0f || cos_t <= 0.0f || dist2 < 1e-6f) return 0.0f;

    float3 wo_s = (s_idx == 0) ? normalize(camera_pos - ev.pos) : normalize(eye_path[s_idx - 1].pos - ev.pos);
    float3 wo_t = (t_idx == 0) ? normalize(lv.normal) : normalize(light_path[t_idx - 1].pos - lv.pos);

    float pdf_omega_s = fmaxf(bsdf_pdf(ev.mtl, wo_s, dir_e_to_l, ns), 1e-6f);
    float pdf_omega_t = fmaxf(bsdf_pdf(lv.mtl, wo_t, dir_e_to_l * -1.0f, nt), 1e-6f);

    float pdf_s_to_t = pdf_omega_s * cos_t / dist2;
    float pdf_t_to_s = pdf_omega_t * cos_s / dist2;
    float sum_ratios = 1.0f;

    float current_ratio = 1.0f;
    float prev_pdf_rev = pdf_t_to_s;

    for(int i = s_idx; i > 0; --i){
        if(eye_path[i].mtl.eta > 0.0f) break; // 碰到完美玻璃/水則截斷
        current_ratio *= prev_pdf_rev / fmaxf(eye_path[i].pdf_fwd, 1e-8f);
        sum_ratios += current_ratio;
        prev_pdf_rev = eye_path[i].pdf_rev;
    }

    current_ratio = 1.0f;
    prev_pdf_rev = pdf_s_to_t;

    for(int i = t_idx; i > 0; --i){
        if(light_path[i].is_light_source){
            current_ratio *= prev_pdf_rev / fmaxf(light_path[i].pdf_fwd, 1e-8f);
            sum_ratios += current_ratio;
            break;
        }
        if(light_path[i].mtl.eta > 0.0f) break;
        current_ratio *= prev_pdf_rev / fmaxf(light_path[i].pdf_fwd, 1e-8f);
        sum_ratios += current_ratio;
        prev_pdf_rev = light_path[i].pdf_rev;
    }

    if(std::isnan(sum_ratios) || std::isinf(sum_ratios) || sum_ratios <= 0.0f) return 0.0f;
    return 1.0f / sum_ratios;
}


// =========================================================
// 核心：CPU 雙向路徑追蹤主程式
// =========================================================
void run_cpu_bdpt(
    const Camera &camera, std::map<int, AABB> &groups, const std::vector<CudaLight> &lights,
    float3 *image_buffer, int eye_depth, int light_depth, int W, int H, int spp, int spl
){
    int total_lights = lights.size();
    if(total_lights == 0) return;

    // 計算場景 Bounding Box (平行光發射用)
    glm::vec3 c_min(1e9f), c_max(-1e9f);
    for(auto &g : groups){
        c_min = glm::min(c_min, g.second.min);
        c_max = glm::max(c_max, g.second.max);
    }
    float3 min_bound = make_float3(c_min.x, c_min.y, c_min.z);
    float3 max_bound = make_float3(c_max.x, c_max.y, c_max.z);

    // 相機座標計算
    float aspect = float(W) / float(H);
    float theta = camera.fov * PI / 180.0f;
    float half_height = std::tan(theta / 2.0f);
    float half_width = aspect * half_height;
    glm::vec3 cw = glm::normalize(camera.eye - camera.look_at);
    glm::vec3 cu = glm::normalize(glm::cross(camera.view_up, cw));
    glm::vec3 cv = glm::cross(cw, cu);
    glm::vec3 cUL = camera.eye - half_width * cu + half_height * cv - cw;
    glm::vec3 cdx = (2.0f * half_width * cu) / float(W);
    glm::vec3 cdy = (-2.0f * half_height * cv) / float(H);
    float3 cam_eye = make_float3(camera.eye.x, camera.eye.y, camera.eye.z);

    int total_light_paths = total_lights * spl;
    std::vector<CudaLightVertex> light_vertices(total_light_paths * light_depth);

    std::cout << "[CPU BDPT] Tracing " << total_light_paths << " Light Paths...\n";

    // -----------------------------------------------------
    // 1. Light Trace (OpenMP)
    // -----------------------------------------------------
    auto cpu_render_start = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        std::mt19937 rng(1337 + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        auto light_trace_start = std::chrono::steady_clock::now();
#pragma omp for schedule(dynamic, 64)
        for(int idx = 0; idx < total_light_paths; ++idx){
            int light_idx = idx % total_lights;
            CudaLight light = lights[light_idx];
            int path_base_idx = idx * light_depth;

            float3 lightray_point, lightray_dir;
            float ray_refract = 1.0f;

            // 光源發射採樣
            if(light.is_parallel){
                lightray_dir = normalize(light.dir);
                float3 scene_center = (min_bound + max_bound) * 0.5f;
                float scene_radius = length(max_bound - min_bound) * 0.5f;
                float3 w = lightray_dir;
                float3 u_vec = (fabs(w.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
                float3 v_vec = normalize(cross(w, u_vec));
                u_vec = normalize(cross(v_vec, w));

                float r1 = dist(rng), r2 = dist(rng);
                float offset_u = (r1 - 0.5f) * scene_radius * 2.0f;
                float offset_v = (r2 - 0.5f) * scene_radius * 2.0f;
                lightray_point = scene_center - lightray_dir * (scene_radius * 2.0f) + u_vec * offset_u + v_vec * offset_v;
            }
            else{
                lightray_point = light.pos;
                float3 w = normalize(light.dir);
                float3 u_vec = (fabs(w.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
                float3 v_vec = normalize(cross(w, u_vec));
                u_vec = normalize(cross(v_vec, w));

                float u1 = dist(rng), u2 = dist(rng);
                float theta = acosf(1.0f - u1 * (1.0f - cosf(light.cutoff)));
                float phi = 2.0f * PI * u2;
                float3 local_dir = make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
                lightray_dir = normalize(u_vec * local_dir.x + v_vec * local_dir.y + w * local_dir.z);
                lightray_point = lightray_point + lightray_dir * light.light_ball.r;
            }

            float3 throughput = light.illum / fmaxf((float) spl, 1.0f);

            CudaLightVertex &vertex0 = light_vertices[path_base_idx];
            vertex0.pos = lightray_point;
            vertex0.normal = lightray_dir;
            vertex0.throughput = throughput;
            vertex0.is_light_source = true;
            vertex0.source_cutoff = light.cutoff;
            vertex0.is_parallel = light.is_parallel;

            float3 last_normal = lightray_dir;
            float3 last_pos = lightray_point;
            float last_pdf_omega = 1.0f / PI;

            // 光線彈射
            for(int depth = 1; depth < light_depth; depth++){
                CudaLightVertex &vertex = light_vertices[path_base_idx + depth];
                vertex.throughput = make_float3(0, 0, 0); // 重置無效狀態

                CudaHit hit = cpu_find_closest_hit(lightray_point, lightray_dir, groups, lights);
                if(!hit.hit) break;
                if(hit.is_light){
                    vertex.pos = hit.pos; vertex.normal = hit.normal; vertex.throughput = throughput;
                    vertex.mtl = hit.mtl; vertex.is_light_source = true; vertex.source_cutoff = 0.0f;
                    vertex.is_parallel = false;
                    break;
                }
                if(length(throughput) < 1e-4f) break;

                float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
                if(dist2 < 1e-6f) break;

                float cos_at_hit = fabs(dot(hit.normal, lightray_dir * -1.0f));
                float cos_at_prev = fabs(dot(last_normal, lightray_dir));
                float pdf_fwd = last_pdf_omega * cos_at_hit / dist2;

                float3 wo = lightray_dir * -1.0f;
                float3 wi, bsdf_val;
                float pdf_omega, new_eta;
                bool is_delta;

                float u_rr = dist(rng), u1 = dist(rng), u2 = dist(rng);
                bsdf_sample(hit.mtl, wo, hit.normal, u_rr, u1, u2, ray_refract, wi, bsdf_val, pdf_omega, is_delta, new_eta);

                if(pdf_omega <= 0.0f && !is_delta) break;

                if(is_delta){
                    throughput = throughput * bsdf_val;
                    lightray_dir = wi; ray_refract = new_eta;
                    lightray_point = hit.pos + hit.normal * (dot(wi, hit.normal) < 0.0f ? -EPSILON : EPSILON);
                    depth--;
                    continue;
                }

                vertex.pos = hit.pos; vertex.normal = hit.normal; vertex.throughput = throughput;
                vertex.mtl = hit.mtl; vertex.is_light_source = false;

                float pdf_rev_omega = bsdf_pdf(hit.mtl, wi, wo, hit.normal);
                float pdf_rev = pdf_rev_omega * cos_at_prev / dist2;

                vertex.pdf_fwd = pdf_fwd; vertex.pdf_rev = pdf_rev;
                throughput = throughput * bsdf_val * fabs(dot(hit.normal, wi)) / pdf_omega;

                if(!cpu_is_valid_color(throughput)) break;

                lightray_dir = wi;
                lightray_point = hit.pos + hit.normal * EPSILON;
                last_pdf_omega = pdf_omega; last_normal = hit.normal; last_pos = hit.pos;
            }
        }
        auto light_trace_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> light_trace_duration = light_trace_end - light_trace_start;
        std::cout << "[CPU BDPT] Light Tracing Completed in " << light_trace_duration.count() << " seconds.\n";
    }

    std::cout << "[CPU BDPT] Tracing Eye Paths & Connecting...\n";

    // -----------------------------------------------------
    // 2. Eye Trace & Connection (OpenMP)
    // -----------------------------------------------------
#pragma omp parallel
    {
        std::mt19937 rng(9999 + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<CudaEyeVertex> local_eye_path(eye_depth);

        auto eye_trace_start = std::chrono::steady_clock::now();
#pragma omp for schedule(dynamic, 16)
        for(int idx = 0; idx < W * H; ++idx){
            int px = idx % W;
            int py = idx / W;
            float3 pixel_accum_color = make_float3(0.0f, 0.0f, 0.0f);

            for(int s = 0; s < spp; ++s){
                float pixel_x = (float) px + dist(rng);
                float pixel_y = (float) py + dist(rng);

                glm::vec3 pixel_pos_glm = cUL + cdx * pixel_x + cdy * pixel_y;
                float3 eyeray_point = cam_eye;
                float3 eyeray_dir = normalize(make_float3(pixel_pos_glm.x, pixel_pos_glm.y, pixel_pos_glm.z) - cam_eye);

                float eyeray_refract = 1.0f;
                float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
                float3 last_normal = eyeray_dir;
                float3 last_pos = cam_eye;
                float last_pdf_omega = 1.0f;

                float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

                for(int depth = 0; depth < eye_depth; depth++){
                    CudaEyeVertex &vertex = local_eye_path[depth];
                    vertex.throughput = make_float3(0, 0, 0);

                    CudaHit hit = cpu_find_closest_hit(eyeray_point, eyeray_dir, groups, lights);
                    if(!hit.hit) break;

                    if(hit.is_light && depth == 0){
                        final_color = final_color + hit.mtl.base_color;
                        break;
                    }

                    float pdf_fwd = 1.0f;
                    if(depth > 0){
                        float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
                        float cos_at_hit = fabs(dot(hit.normal, eyeray_dir * -1.0f));
                        pdf_fwd = last_pdf_omega * cos_at_hit / fmaxf(dist2, 1e-6f);
                    }

                    vertex.pos = hit.pos; vertex.normal = hit.normal; vertex.throughput = throughput;
                    vertex.mtl = hit.mtl; vertex.pdf_fwd = 0.0f; vertex.pdf_rev = 1.0f;

                    // == 連線測試 (Connection) ==
                    float3 total_L = make_float3(0.0f, 0.0f, 0.0f);
                    for(int light_idx = 0; light_idx < total_light_paths * light_depth; light_idx++){
                        const CudaLightVertex &lv = light_vertices[light_idx];
                        if(length(lv.throughput) < 1e-6f) continue;

                        float3 d_vec = lv.pos - vertex.pos;
                        float dist2 = dot(d_vec, d_vec);
                        if(dist2 < 1e-6f) continue;

                        float dist = sqrtf(dist2);
                        float3 wi = d_vec / dist;

                        float cosE = fmaxf(0.0f, dot(vertex.normal, wi));
                        float cosL = fmaxf(0.0f, dot(lv.normal, wi * -1.0f));
                        if(cosE <= 0.0f || cosL <= 0.0f) continue;

                        int current_t_idx = light_idx % light_depth;
                        if(lv.is_light_source && lv.source_cutoff > 0.0f && !lv.is_parallel){
                            int real_light_idx = (light_idx / light_depth) % total_lights;
                            float3 light_dir = normalize(make_float3(lights[real_light_idx].dir.x, lights[real_light_idx].dir.y, lights[real_light_idx].dir.z));
                            if(dot(light_dir, wi * -1.0f) < cosf(lv.source_cutoff)) continue;
                        }

                        float3 wo_e = eyeray_dir * -1.0f;
                        float3 fE = bsdf_evaluate(vertex.mtl, wo_e, wi, vertex.normal);
                        float3 fL = make_float3(1.0f, 1.0f, 1.0f);

                        if(!lv.is_light_source && current_t_idx > 0){
                            float3 prev_lv_pos = light_vertices[light_idx - 1].pos;
                            float3 wo_l = normalize(prev_lv_pos - lv.pos);
                            fL = bsdf_evaluate(lv.mtl, wo_l, wi * -1.0f, lv.normal);
                        }

                        if((fE.x <= 0.0f && fE.y <= 0.0f && fE.z <= 0.0f) ||
                            (fL.x <= 0.0f && fL.y <= 0.0f && fL.z <= 0.0f)) continue;

                        float3 transmittance = cpu_check_visibility(vertex.pos + vertex.normal * EPSILON, lv.pos + lv.normal * EPSILON, groups);
                        if(transmittance.x <= 0.0f && transmittance.y <= 0.0f && transmittance.z <= 0.0f) continue;

                        float G = (cosE * cosL) / fmaxf(dist2, 1e-4f);

                        // 計算 MIS 權重
                        const CudaLightVertex *current_light_path_base = &light_vertices[(light_idx / light_depth) * light_depth];
                        float mis_w = cpu_calculate_mis_weight(local_eye_path.data(), depth, current_light_path_base, current_t_idx, d_vec, dist2, cam_eye);

                        float3 contrib = vertex.throughput * fE * G * fL * lv.throughput * transmittance * mis_w;

                        if(cpu_is_valid_color(contrib)){
                            contrib = cpu_clamp_radiance(contrib, 15.0f);
                            total_L = total_L + contrib;
                        }
                    }
                    final_color = final_color + total_L;

                    // == Eye Path 彈射 ==
                    float3 wo = eyeray_dir * -1.0f;
                    float3 wi, bsdf_val;
                    float pdf_omega, new_eta;
                    bool is_delta;

                    float u_rr = dist(rng), u1 = dist(rng), u2 = dist(rng);
                    bsdf_sample(hit.mtl, wo, hit.normal, u_rr, u1, u2, eyeray_refract, wi, bsdf_val, pdf_omega, is_delta, new_eta);

                    if(pdf_omega <= 0.0f && !is_delta) break;

                    if(is_delta){
                        throughput = throughput * bsdf_val;
                        eyeray_dir = wi; eyeray_refract = new_eta;
                        eyeray_point = hit.pos + hit.normal * (dot(wi, hit.normal) < 0.0f ? -EPSILON : EPSILON);
                        last_pos = hit.pos; last_normal = hit.normal; last_pdf_omega = 1.0f;
                        depth--; continue;
                    }

                    float pdf_rev_omega = bsdf_pdf(hit.mtl, wi, wo, hit.normal);
                    float dist2 = dot(hit.pos - last_pos, hit.pos - last_pos);
                    float cos_at_prev = fabs(dot(last_normal, eyeray_dir));
                    vertex.pdf_fwd = pdf_fwd;
                    vertex.pdf_rev = pdf_rev_omega * cos_at_prev / fmaxf(dist2, 1e-6f);

                    throughput = throughput * bsdf_val * fabs(dot(hit.normal, wi)) / pdf_omega;
                    if(!cpu_is_valid_color(throughput)) break;

                    eyeray_dir = wi;
                    eyeray_point = hit.pos + hit.normal * EPSILON;
                    last_pdf_omega = pdf_omega; last_normal = hit.normal; last_pos = hit.pos;
                }

                if(!cpu_is_valid_color(final_color)) final_color = make_float3(0.0f, 0.0f, 0.0f);
                pixel_accum_color = pixel_accum_color + final_color;
            }

            image_buffer[idx] = pixel_accum_color / (float) spp;
        }
        auto eye_trace_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> eye_trace_duration = eye_trace_end - eye_trace_start;
        std::cout << "[CPU BDPT] Eye Tracing & Connection Completed in " << eye_trace_duration.count() << " seconds.\n";
    }
    auto cpu_render_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_duration = cpu_render_end - cpu_render_start;
    std::cout << "[CPU BDPT] Total Rendering Time in " << total_duration.count() << " seconds.\n";
}