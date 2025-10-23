#include "rt.h"
#include <map>
#include <iostream>


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
}

inline Hit first_hit(const Ray &inRay, const std::map<int, AABB> &groups,
    float tMin = 1e-4f, float tMax = std::numeric_limits<float>::infinity()){
    Ray ray = inRay;
    ray.vec = glm::normalize(ray.vec);

    Hit best;
    for(const auto g : groups){
        AABB box = g.second;
        if(box.intersectAABB(ray)){
            std::vector<Object *> scene = box.objs;
            for(const Object *obj : scene){
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

glm::vec3 phong(const Hit &hit, const std::vector<Light> &lights,
    const std::map<int, AABB> &groups){
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
    const std::map<int, AABB> &groups,
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
        Ray reflect_ray = Ray(h.pos, reflect_vec);
        reflect_color += path_tracing(reflect_ray, groups, lights, depth + 1);
        color = color + reflect_color * h.obj->mtl.reflect;
    }

    // std::cout << color << std::endl;
    return color;
}

