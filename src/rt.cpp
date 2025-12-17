#include "rt.h"
#include <map>
#include <iostream>
// #include <corecrt_math_defines.h>
#include <omp.h>

#define LIGHT_COLOR (glm::vec3(0.7f,  .7f, .7f))
#define LIGHT_POS (glm::vec3(0, 0.49, 0.2))
#define LIGHT_R 0.25f
#define LIGHT_SAMPLE 8000

std::vector<LightVertex> light_subpath;
std::vector<std::vector<std::vector<EyeVertex> > > screen_info;
std::vector<Light> light;
std::map<int, AABB> light_groups, eye_groups;


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
}

// 輔助函式：將 glm::vec3 轉為 CudaVec3
CudaVec3 to_cv3(const glm::vec3 &v){
    return { v.x, v.y, v.z };
}

// 輔助函式：將 Material 轉為 CudaMaterial
CudaMaterial to_cmtl(const Material &m){
    CudaMaterial cm;
    cm.Kd = to_cv3(m.Kd);
    cm.refract = m.refract;
    return cm;
}

std::vector<glm::vec3> run_cuda_eye_light_connect(
    int W, int H,
    std::map<int, AABB> &groups
){
    // 1. Flatten Light Path
    std::vector<CudaLightVertex> h_light_path;
    h_light_path.reserve(light_subpath.size());
    for(const auto &lv : light_subpath){
        CudaLightVertex clv;
        clv.pos = to_cv3(lv.pos);
        clv.normal = to_cv3(lv.normal);
        clv.throughput = to_cv3(lv.throughput);
        if(lv.obj) clv.mtl = to_cmtl(lv.obj->mtl);
        else{ // Light source
            clv.mtl.Kd = { 1,1,1 };
            clv.mtl.refract = 0;
        }
        h_light_path.push_back(clv);
    }

    // 2. Flatten Eye Paths & Generate Offsets
    std::vector<CudaEyeVertex> h_eye_paths_flat;
    std::vector<int> h_eye_offsets(W * H);
    std::vector<int> h_eye_counts(W * H);

    int current_offset = 0;
    for(int j = 0; j < H; j++){
        for(int i = 0; i < W; i++){
            int idx = j * W + i; // 注意：要配合 render_array 的邏輯，或直接線性
            // 由於 screen_info 是 [W][H]，我們需要確保索引一致
            // main.cpp 的寫法是 screen_info[i][j] (col, row)
            // CUDA kernel 是一維陣列，我們假設 row-major (j*W + i)

            const auto &path = screen_info[i][j];
            h_eye_offsets[idx] = current_offset;
            h_eye_counts[idx] = path.size();

            for(const auto &ev : path){
                CudaEyeVertex cev;
                cev.pos = to_cv3(ev.pos);
                cev.normal = to_cv3(ev.normal);
                cev.throughput = to_cv3(ev.throughput);
                if(ev.obj) cev.mtl = to_cmtl(ev.obj->mtl);
                else cev.mtl = { {0,0,0}, 0 };
                h_eye_paths_flat.push_back(cev);
            }
            current_offset += path.size();
        }
    }

    // 3. Flatten Scene Objects (Simple Iteration over Groups)
    std::vector<CudaSphere> h_spheres;
    std::vector<CudaTriangle> h_triangles;

    for(auto &g : groups){
        for(Object *obj : g.second.objs){
            if(Sphere *s = dynamic_cast<Sphere *>(obj)){
                CudaSphere cs;
                cs.center = to_cv3(s->center);
                cs.r = s->r;
                cs.mtl = to_cmtl(s->mtl);
                cs.id = s->obj_id;
                h_spheres.push_back(cs);
            }
            else if(Triangle *t = dynamic_cast<Triangle *>(obj)){
                CudaTriangle ct;
                ct.v0 = to_cv3(t->vert[0]);
                ct.v1 = to_cv3(t->vert[1]);
                ct.v2 = to_cv3(t->vert[2]);
                ct.mtl = to_cmtl(t->mtl);
                ct.id = t->obj_id;
                h_triangles.push_back(ct);
            }
        }
    }

    // 4. Output Buffer
    std::vector<CudaVec3> cuda_output(W * H);

    // 5. Call CUDA Wrapper
    cuda_eye_light_connect_wrapper(
        W, H,
        h_light_path.data(), h_light_path.size(),
        h_eye_paths_flat.data(),
        h_eye_offsets.data(), h_eye_counts.data(),
        h_spheres.data(), h_spheres.size(),
        h_triangles.data(), h_triangles.size(),
        to_cv3(LIGHT_COLOR), // 假設 LIGHT_COLOR 在 rt.h 有定義
        cuda_output.data(),
        TRACE_MODE // connect_mode = 1 for eye-light connection
    );

    // 6. Convert back to glm::vec3 for existing pipeline
    std::vector<glm::vec3> result(W * H);
    for(int k = 0; k < W * H; k++){
        result[k] = glm::vec3(cuda_output[k].x, cuda_output[k].y, cuda_output[k].z);
    }
    return result;
}

inline Hit first_hit(Ray &inRay, std::map<int, AABB> &groups,
    float tMin = 1e-4f, float tMax = std::numeric_limits<float>::infinity()){
    Ray ray = inRay;
    ray.vec = glm::normalize(ray.vec);

    Hit best;
    for(auto &g : groups){
        AABB *box = &(g.second);
        if(box->intersectAABB(ray)){
            std::vector<Object *> *scene = &(box->objs);
            for(Object *obj : *scene){
                float t, u, v;
                if(!obj->check_intersect(ray, t, u, v, tMin, tMax)) continue;
                if(t < best.t){
                    best.hit = true;
                    best.t = t;
                    best.pos = ray.point + t * ray.vec;
                    best.u = u;
                    best.v = v;
                    best.eye_dir = glm::normalize(ray.vec);
                    best.obj = obj;

                    if(dynamic_cast<const Sphere *>(obj))   best.kind = ObjKind::Sphere;
                    else if(dynamic_cast<const Triangle *>(obj)) best.kind = ObjKind::Triangle;
                    else best.kind = ObjKind::Unknown;

                    // 法向
                    best.normal = obj->normal_at(best.pos, ray, u, v);

                    // 重心 or UV
                    if(best.kind == ObjKind::Triangle){
                        best.bary = glm::vec3(1.0f - u - v, u, v); // (w,u,v)
                    }
                    else if(best.kind == ObjKind::Sphere){
                        best.bary = glm::vec3(u, v, 0.0f);
                    }
                }
            }
        }
    }
    return best;
}


void init_lightray(std::map<int, AABB> &groups){
    light_subpath.clear();
    for(int i = 0; i < LIGHT_SAMPLE; i++){
        glm::vec3 light_pos = LIGHT_POS + glm::vec3(
            random_float(-1, 1) * LIGHT_R, 0, random_float(-1, 1) * LIGHT_R);
        glm::vec3 light_dir = sample_hemisphere_cosine(glm::vec3(0, -1, 0));
        glm::vec3 accumulated = lightray_tracer(Ray(light_pos, light_dir, 1, RayType::LIGHT), groups, (LIGHT_COLOR * 120.f / (float) LIGHT_SAMPLE));

        // std::cout << light_pos << " " << light_dir << std::endl;

    }
    std::cout << "Light Ray：" << light_subpath.size() << std::endl;
}

//rad
inline float get_theta(glm::vec3 a, glm::vec3 b){
    return std::acos(glm::dot(a, b) / (glm::length(a) * glm::length(b)));
}

glm::vec3 sample_hemisphere_cosine(const glm::vec3 &normal){
    glm::vec3 N = glm::normalize(normal);
    glm::vec3 T, B;

    if(std::fabs(N.z) < 0.999f){
        T = glm::normalize(glm::cross(glm::vec3(0, 0, 1), N));
    }
    else{
        T = glm::normalize(glm::cross(glm::vec3(0, 1, 0), N));
    }
    B = glm::cross(N, T);

    float u1 = random_float(0, 1);
    float u2 = random_float(0, 1);

    float r = std::sqrt(u1);
    float phi = 2.0f * float(M_PI) * u2;

    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1)); // cosθ

    glm::vec3 local(x, y, z);
    glm::vec3 world = x * T + y * B + z * N;
    return glm::normalize(world);
}

// 產生單位球內的隨機向量 (用於 Glossy Jitter)
glm::vec3 random_in_unit_sphere(){
    while(true){
        glm::vec3 p = glm::vec3(random_float(-1, 1), random_float(-1, 1), random_float(-1, 1));
        if(glm::length(p) >= 1.0f) continue;
        return p;
    }
}

glm::vec3 lightray_tracer(Ray light_ray,
    std::map<int, AABB> &groups,
    glm::vec3 throughput){
    glm::vec3 accumulated(0.0f);

    // std::cout << throughput << std::endl;
    LightVertex src;
    src.pos = light_ray.point;
    src.normal = light_ray.vec;
    src.wi = -light_ray.vec;
    src.throughput = throughput;
    src.obj = nullptr;

    light_subpath.push_back(src);

    for(int depth = 0; depth < LIGHT_DEPTH; depth++){

        Hit h = first_hit(light_ray, groups);
        if(!h.hit) break;
        glm::vec3 n = glm::normalize(h.normal);

        float rn = rng_uniform01();
        if(h.obj->mtl.reflect > 0 && rn <= h.obj->mtl.reflect){
            glm::vec3 I = glm::normalize(light_ray.vec);
            glm::vec3 N = n;

            glm::vec3 R = glm::reflect(I, N);

            light_ray.vec = R;
            light_ray.point = h.pos + light_ray.vec * 1e-4f;
            depth--;
            continue;
        }

        if(h.obj->mtl.refract > 0){
            glm::vec3 I = glm::normalize(light_ray.vec);
            glm::vec3 N = n;

            float n1 = light_ray.refract;
            float n2 = h.obj->mtl.refract;

            float cosNI = glm::dot(I, N);

            if(cosNI > 0.0f){
                std::swap(n1, n2);
                N = -N;
                cosNI = glm::dot(I, N);
            }

            float eta = n1 / n2;
            glm::vec3 T = glm::refract(I, N, eta);

            if(glm::length(T) < 1e-6f){
                glm::vec3 R = glm::reflect(I, N);
                light_ray.vec = glm::normalize(R);
            }
            else{
                light_ray.vec = glm::normalize(T);
                light_ray.refract = n2;
            }

            light_ray.point = h.pos + light_ray.vec * 1e-4f;

            depth--;
            continue;
        }

        float diff_glos = rng_uniform01();
        if(diff_glos <= h.obj->mtl.glossy){
            glm::vec3 I = glm::normalize(light_ray.vec);
            glm::vec3 N = n;
            glm::vec3 R = glm::reflect(I, N);

            // 計算粗糙度
            float roughness = (h.obj->mtl.exp > 1000.0f) ? 0.0f : (1.0f / (h.obj->mtl.exp * 0.05f + 0.001f));

            glm::vec3 jitter = random_in_unit_sphere() * roughness;
            glm::vec3 new_dir = glm::normalize(R + jitter);

            if(glm::dot(new_dir, N) <= 0.0f){
                new_dir = new_dir - 2.0f * glm::dot(new_dir, N) * N;
            }

            new_dir = glm::normalize(new_dir);

            light_ray.vec = new_dir;
            light_ray.point = h.pos + light_ray.vec * 1e-4f;
            depth--;
            continue;
        }
        else{
            throughput *= h.obj->mtl.Kd;

            LightVertex v;
            v.pos = h.pos;
            v.normal = n;
            v.wi = -light_ray.vec;
            v.throughput = throughput;
            v.obj = h.obj;
            light_subpath.push_back(v);

            accumulated += throughput;

            glm::vec3 newDir = sample_hemisphere_cosine(n);
            light_ray = Ray(h.pos + newDir * 1e-4f, newDir, 1, RayType::LIGHT);
        }
    }

    return accumulated;
}

void init_light_group(){
    std::vector<Sphere *> light_vec;
    for(auto &i : light_subpath){
        Sphere *s = new Sphere();
        s->center = i.pos;
        Material *mtl = new Material();
        mtl->Kd = i.throughput;
        s->mtl = *mtl;
        // std::cout << i.throughput << std::endl;
        s->r = 0.01;
        light_vec.push_back(s);
    }

    AABB light_box;
    for(int i = 0; i < light_vec.size(); i++)
        light_box.add_obj(light_vec[i]);
    light_groups[0] = light_box;
    std::cout << "Light Vertexs: " << light_vec.size() << std::endl;

    std::vector<Sphere *> eye_vec;
    for(auto &i : screen_info){
        for(auto &j : i){
            for(auto &r : j){
                Sphere *s = new Sphere();
                s->center = r.pos;
                Material *mtl = new Material();
                mtl->Kd = r.throughput;
                // std::cout << s->center << std::endl;
                s->mtl = *mtl;
                // std::cout << i.throughput << std::endl;
                s->r = 0.01;
                eye_vec.push_back(s);
            }
        }
    }

    AABB eye_box;
    for(int i = 0; i < eye_vec.size(); i++)
        eye_box.add_obj(eye_vec[i]);
    eye_groups[0] = eye_box;
    std::cout << "Eye Vertexs: " << eye_vec.size() << std::endl;


}

void resize_screen_info(int W, int H){
    screen_info.resize(W, std::vector<std::vector<EyeVertex> >(H));
}

void init_eyeray(std::map<int, AABB> &groups, std::vector<EyeRayInfo> &eye_ray,
    int W, int H){
    int eyeray_cnt = 0;
#pragma omp parallel for schedule(dynamic, 128)
    for(int i = 0; i < eye_ray.size(); i++){
        screen_info[eye_ray[i].i][eye_ray[i].j].clear();
        std::vector<EyeVertex> res = eyeray_tracer(eye_ray[i].ray, groups, glm::vec3(1, 1, 1));
        screen_info[eye_ray[i].i][eye_ray[i].j] = res;
        eyeray_cnt += res.size();
    }
    std::cout << "Eye Ray：" << eyeray_cnt << std::endl;
}

std::vector<EyeVertex> eyeray_tracer(Ray &eye_ray,
    std::map<int, AABB> &groups,
    glm::vec3 throughput){

    std::vector<EyeVertex> ret;
    for(int depth = 0; depth < EYE_DEPTH; depth++){

        Hit h = first_hit(eye_ray, groups);
        if(!h.hit) break;
        glm::vec3 n = glm::normalize(h.normal);

        float rn = rng_uniform01();
        if(h.obj->mtl.reflect > 0 && rn <= h.obj->mtl.reflect){
            glm::vec3 I = glm::normalize(eye_ray.vec);
            glm::vec3 N = n;

            glm::vec3 R = glm::reflect(I, N);

            eye_ray.vec = R;
            eye_ray.point = h.pos + eye_ray.vec * 1e-4f;
            depth--;
            continue;
        }

        if(h.obj->mtl.refract > 0){
            glm::vec3 I = glm::normalize(eye_ray.vec);
            glm::vec3 N = n;

            float n1 = eye_ray.refract;
            float n2 = h.obj->mtl.refract;

            float cosNI = glm::dot(I, N);

            if(cosNI > 0.0f){
                std::swap(n1, n2);
                N = -N;
                cosNI = glm::dot(I, N);
            }

            float eta = n1 / n2;
            glm::vec3 T = glm::refract(I, N, eta);

            if(glm::length(T) < 1e-6f){
                glm::vec3 R = glm::reflect(I, N);
                eye_ray.vec = glm::normalize(R);
            }
            else{
                eye_ray.vec = glm::normalize(T);
                eye_ray.refract = n2;
            }

            eye_ray.point = h.pos + eye_ray.vec * 1e-4f;

            depth--;
            continue;
        }

        float diff_glos = rng_uniform01();
        if(diff_glos <= h.obj->mtl.glossy){
            glm::vec3 I = glm::normalize(eye_ray.vec);
            glm::vec3 N = n;
            glm::vec3 R = glm::reflect(I, N);

            float roughness = (h.obj->mtl.exp > 1000.0f) ? 0.0f : (1.0f / (h.obj->mtl.exp * 0.05f + 0.001f));

            glm::vec3 jitter = random_in_unit_sphere() * roughness;
            glm::vec3 new_dir = glm::normalize(R + jitter);

            if(glm::dot(new_dir, N) <= 0.0f){
                new_dir = new_dir - 2.0f * glm::dot(new_dir, N) * N;
            }

            new_dir = glm::normalize(new_dir);

            eye_ray.vec = new_dir;
            eye_ray.point = h.pos + eye_ray.vec * 1e-4f;
            depth--;
            continue;
        }
        else{
            throughput *= h.obj->mtl.Kd;

            EyeVertex v;
            v.pos = h.pos;
            v.normal = n;
            v.wi = -eye_ray.vec;
            v.throughput = throughput;
            v.obj = h.obj;
            ret.push_back(v);

            glm::vec3 newDir = sample_hemisphere_cosine(n);
            eye_ray = Ray(h.pos + newDir * 1e-4f, newDir, 1, RayType::EYE);
        }
    }
    return ret;
}

glm::vec3 eye_light_connect(int i, int j, std::map<int, AABB> &groups){
    const std::vector<EyeVertex> &eye_subpath = screen_info[i][j];
    const std::vector<LightVertex> &light_path = light_subpath;

    if(eye_subpath.empty() || light_path.empty()){
        return glm::vec3(0.0f);
    }

    glm::vec3 L(0.0f);

    for(const auto &ev : eye_subpath){
        glm::vec3 nE = glm::normalize(ev.normal);
        glm::vec3 KdE = ev.obj->mtl.Kd;
        glm::vec3 fE = KdE * (1.0f / float(M_PI));  // Lambert BRDF
        for(const auto &lv : light_path){
            glm::vec3 nL = glm::normalize(lv.normal);
            glm::vec3 KdL = LIGHT_COLOR;
            glm::vec3 Le = (lv.obj == nullptr) ? LIGHT_COLOR : glm::vec3(1.0f);
            glm::vec3 d = lv.pos - ev.pos;
            float dist2 = glm::dot(d, d);
            if(dist2 <= 1e-8f) continue;

            float dist = std::sqrt(dist2);
            glm::vec3 wi = d / dist;

            float cosE = glm::max(0.0f, glm::dot(nE, wi));
            float cosL = glm::max(0.0f, glm::dot(nL, -wi));

            if(cosE <= 0.0f || cosL <= 0.0f) continue;

            glm::vec3 transmittance(1.0f);
            bool occluded = false;

            Ray shadow_ray(ev.pos + wi * 1e-4f, wi, 1, RayType::LIGHT);
            float maxDist = dist - 1e-4f;

            while(true){
                Hit shadow_hit = first_hit(shadow_ray, groups, 1e-4f, maxDist);
                if(!shadow_hit.hit){
                    break;
                }

                const Material &mtlS = shadow_hit.obj->mtl;

                if(mtlS.refract <= 0.0f){

                    occluded = true;
                    break;
                }
                else{

                    shadow_ray.point = shadow_hit.pos + shadow_ray.vec * 1e-4f;
                    continue;
                }
            }

            if(occluded) continue;

            float G = (cosE * cosL) / dist2;

            glm::vec3 contrib = ev.throughput * fE;
            contrib *= G;
            contrib *= lv.throughput;
            contrib *= transmittance;

            L += contrib;
        }
    }

    return L;
}

glm::vec3 light_debuger(int i, int j, glm::vec3 UL, glm::vec3 dx, glm::vec3 dy, std::map<int, AABB> &groups, Camera ori_cam){
    glm::vec3 col(0);
    for(int k = 0; k < SAMPLE; k++){
        // std::cout << ori_cam.fov << std::endl;
        float jx = rng_uniform01() - 0.5f;
        float jy = rng_uniform01() - 0.5f;
        glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f + jx) + dy * (float(j) + 0.5f + jy);

        glm::vec3 ray_dir = glm::normalize(pixel_pos - ori_cam.eye);
        Ray ray(ori_cam.eye, ray_dir, 0, RayType::EYE);

        // col += path_tracing(ray, eye_groups, light, 0);
        col += path_tracing(ray, light_groups, light, 0);

    }
    return col / SAMPLE;
}

glm::vec3 phong(Hit &hit, std::vector<Light> &lights,
    std::map<int, AABB> &groups){
    const Material &mtl = hit.obj->mtl;

    glm::vec3 N = glm::normalize(hit.normal);
    glm::vec3 V = glm::normalize(hit.eye_dir);   // 表面 → 眼睛

    glm::vec3 color = mtl.Kg;

    for(const auto &l : lights){
        glm::vec3 L = glm::normalize(l.dir);

        Ray shadow_ray = Ray();
        shadow_ray.point = hit.pos;
        shadow_ray.vec = l.dir;
        Hit shadow_hit = first_hit(shadow_ray, groups);

        float ndotl = glm::max(0.0f, glm::dot(N, L));
        glm::vec3 diffuse = mtl.Kd * l.illum * ndotl;

        glm::vec3 R = glm::reflect(L, N);
        float rv = glm::max(0.0f, glm::dot(R, V));
        float spec_pow = (mtl.exp > 0.0f) ? std::pow(rv, mtl.exp) : 0.0f;
        glm::vec3 specular = mtl.Ks * l.illum * spec_pow;

        if(!shadow_hit.hit){
            color += diffuse + specular;
        }
    }
    return color;
}

glm::vec3 path_tracing(Ray ray,
    std::map<int, AABB> &groups,
    std::vector<Light> &lights,
    int depth){
    if(depth == MAX_DPETH) return glm::vec3{ 0.0f };
    Hit h = first_hit(ray, groups);
    if(!h.hit) return glm::vec3(0.0f);
    // if(h.kind == ObjKind::Sphere){
    //     std::cout << h.obj->obj_id << std::endl;
    // }
    // if(h.kind == ObjKind::Triangle){
    //     std::cout << h.obj->obj_id << std::endl;
    // }

    glm::vec3 n = glm::normalize(h.normal);
    // glm::vec3 color = glm::vec3(255 * (h.pos.x + 1.0) / 2.0,
    //     255 * (h.pos.y + 1.0) / 2.0,
    //     255 * (h.pos.z + 1.0) / 2.0);
    // std::cout << h.pos << std::endl;
    glm::vec3 color = phong(h, lights, groups);

    float prob = get_rand();

    // if(prob < h.obj->mtl.reflect){
    //     glm::vec3 reflect_color{ 0.0f };
    //     for(int i = 0; i < SAMPLE; i++){
    //         glm::vec3 reflect_vec = glm::reflect(ray.vec, n);
    //         glm::vec3 esp = glm::vec3(get_esp(), get_esp(), get_esp());
    //         reflect_vec += esp;
    //         reflect_vec = glm::normalize(reflect_vec);
    //         Ray reflect_ray = Ray(h.pos, reflect_vec);
    //         reflect_color += path_tracing(reflect_ray, scene, lights, depth + 1);
    //     }
    //     reflect_color = reflect_color / SAMPLE;
    //     color = color + reflect_color;
    // }

    if(h.obj->mtl.reflect > ESP){
        glm::vec3 reflect_color(0.0f);
        glm::vec3 reflect_vec = glm::reflect(ray.vec, n);
        Ray reflect_ray = Ray(h.pos, reflect_vec, 1, RayType::LIGHT);
        reflect_color += path_tracing(reflect_ray, groups, lights, depth + 1);
        color = color + reflect_color * h.obj->mtl.reflect;
    }

    // std::cout << color << std::endl;
    return color;
}

