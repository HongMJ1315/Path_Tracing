#include "rt.h"
#include <map>
#include <iostream>

#define LIGHT_COLOR (glm::vec3( 1.f,  1.f, 1.f))
#define LIGHT_POS (glm::vec3(0, 0.49, 0))
#define LIGHT_R 0.25f
#define LIGHT_SAMPLE 100
#define LIGHT_RAY_SAMPLE 5
#define LIGHT_RAY_RECUSIVE_SAMPLE 3
#define LIGHT_DEPTH 2

std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
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

std::vector<LightVertex> light_subpath;



void init_lightray(std::map<int, AABB> &groups){
    light_subpath.clear();
    for(int i = 0; i < LIGHT_SAMPLE; i++){
        glm::vec3 light_pos = LIGHT_POS + glm::vec3(
            random_float(-1, 1) * LIGHT_R, 0, random_float(-1, 1) * LIGHT_R);
        for(int j = 0; j < LIGHT_RAY_SAMPLE; j++){
            glm::vec3 light_dir = sample_hemisphere_uniform(glm::vec3(0, -1, 0));
            glm::vec3 accumulated = lightray_tracer(Ray(light_pos, light_dir, 1, RayType::LIGHT), groups, LIGHT_COLOR, 0);

            // std::cout << light_pos << " " << light_dir << std::endl;
        }

    }
}

//rad
inline float get_theta(glm::vec3 a, glm::vec3 b){
    return std::acos(glm::dot(a, b) / (glm::length(a) * glm::length(b)));
}

glm::vec3 lightray_tracer(Ray light_ray, std::map<int, AABB> &groups, glm::vec3 throughput, int depth){
    if(depth == LIGHT_DEPTH) return glm::vec3(0, 0, 0);

    glm::vec3 accumulated(0.0f);
    for(int i = 0; i < LIGHT_RAY_RECUSIVE_SAMPLE; i++){
        Hit h = first_hit(light_ray, groups);

        if(!h.hit) break;


        glm::vec3 n = glm::normalize(h.normal);

        throughput *= h.obj->mtl.Kd;

        LightVertex v;
        v.pos = h.pos;
        v.normal = n;
        v.wi = -light_ray.vec;
        v.throughput = throughput * h.obj->mtl.Kd;
        v.obj = h.obj;

        light_subpath.push_back(v);

        // 若你想看顏色，可以累積一下
        accumulated += throughput;
        throughput = throughput / float(LIGHT_RAY_RECUSIVE_SAMPLE);

        // 產生下一次反彈方向（diffuse）
        glm::vec3 newDir = sample_hemisphere_uniform(n);
        Ray nxt_light_ray = Ray(h.pos + newDir * 1e-4f, newDir, 1, RayType::LIGHT);
        glm::vec3 acc = lightray_tracer(nxt_light_ray, groups, throughput, depth + 1);
        accumulated += acc;
    }
    return accumulated;
}


std::vector<Light> light;
std::map<int, AABB> light_groups;


void init_light_group(){
    AABB box;
    std::vector<Sphere *> light_vec;
    for(auto &i : light_subpath){
        Sphere *s = new Sphere();
        s->center = i.pos;
        Material *mtl = new Material();
        mtl->Ka = mtl->Kd = i.throughput;
        s->mtl = *mtl;
        // std::cout << i.throughput << std::endl;
        s->r = 0.01;
        light_vec.push_back(s);
    }

    std::sort(light_vec.begin(), light_vec.end(),
        [](Sphere *a, Sphere *b){
        return a->center.x > b->center.x;
    });

    int split = light_vec.size() / 4;
    for(int i = 0; i < 4; i++){
        AABB t_box;
        for(int j = 0; j < split; j++){
            if(i * split + j > light_vec.size()) break;
            t_box.add_obj(light_vec[i * split + j]);
        }
        light_groups[i] = t_box;
    }
    std::cout << light_vec.size() << std::endl;
    std::vector<Light> light;

}

glm::vec3 bdpt(Ray ray,
    std::map<int, AABB> &groups,
    int depth){

    return path_tracing(ray, light_groups, light, 0);
}


glm::vec3 phong(Hit &hit, std::vector<Light> &lights,
    std::map<int, AABB> &groups){
    const Material &mtl = hit.obj->mtl;

    glm::vec3 N = glm::normalize(hit.normal);
    glm::vec3 V = glm::normalize(hit.eye_dir);   // 表面 → 眼睛

    glm::vec3 color = mtl.Ka;

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

