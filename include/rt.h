#include "object.h"
#include "rng.h"
#include <vector>
#include <map>

#define MAX_DPETH 1
#define SAMPLE 25.f

struct LightVertex{
    glm::vec3 pos;          // 位置
    glm::vec3 normal;          // 法線
    glm::vec3 wi;         // 往「上一個頂點」的方向（入射方向），通常 = -ray.vec
    glm::vec3 throughput; // 從光源一路傳到這個點的能量 (beta / W_hat)
    Object *obj;          // 打到的物體（之後連接時可能需要）
    // 你想加 id / kind 都可以
};


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec);
std::istream &operator>>(std::istream &is, glm::vec3 &vec);

inline Hit first_hit(Ray &inRay, std::map<int, AABB> &groups,
    float tMin, float tMax);
void init_light_group();

glm::vec3 bdpt(Ray ray,
    std::map<int, AABB> &groups,
    int depth);
glm::vec3 next_light_diffuse(glm::vec3 n);
glm::vec3 lightray_tracer(Ray light_ray, std::map<int, AABB> &groups, glm::vec3 throughput, int depth);
void init_lightray(std::map<int, AABB> &groups);
glm::vec3 phong(Hit &hit, std::vector<Light> &lights,
    std::map<int, AABB> &groups);


glm::vec3 path_tracing(Ray ray,
    std::map<int, AABB> &groups,
    std::vector<Light> &lights,
    int depth);