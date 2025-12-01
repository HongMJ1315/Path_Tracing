#include "object.h"
#include "rng.h"
#include <vector>
#include <map>

#define MAX_DPETH 1
#define SAMPLE 5.f

struct LightVertex{
    glm::vec3 pos;          // 位置
    glm::vec3 normal;          // 法線
    glm::vec3 wi;         // 往「上一個頂點」的方向（入射方向），通常 = -ray.vec
    glm::vec3 throughput; // 從光源一路傳到這個點的能量 (beta / W_hat)
    Object *obj;          // 打到的物體（之後連接時可能需要）
    // 你想加 id / kind 都可以
};

struct EyeVertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 wi;         // 入射方向（從這個點看回上一個點 / camera）
    glm::vec3 throughput; // 從 camera 走到這個點的 β
    Object *obj;
};

struct EyeRayInfo{
    int i, j;
    Ray ray;
};

std::ostream &operator<<(std::ostream &os, const glm::vec3 vec);
std::istream &operator>>(std::istream &is, glm::vec3 &vec);

inline Hit first_hit(Ray &inRay, std::map<int, AABB> &groups,
    float tMin, float tMax);

void init_lightray(std::map<int, AABB> &groups);

glm::vec3 sample_hemisphere_cosine(const glm::vec3 &normal);

glm::vec3 lightray_tracer(Ray light_ray, std::map<int, AABB> &groups, glm::vec3 throughput);

void init_light_group();

void init_eyeray(std::map<int, AABB> &groups, std::vector<EyeRayInfo> eyeray,
    int W, int H);

std::vector<EyeVertex> eyeray_tracer(Ray ray,
    std::map<int, AABB> &groups,
    glm::vec3 throughput);

glm::vec3 eye_light_connect(int i, int j, std::map<int, AABB> &groups);

glm::vec3 light_debuger(int i, int j, glm::vec3 UL, glm::vec3 dx, glm::vec3 dy, std::map<int, AABB> &groups, Camera ori_cam);

glm::vec3 phong(Hit &hit, std::vector<Light> &lights,
    std::map<int, AABB> &groups);

glm::vec3 path_tracing(Ray ray,
    std::map<int, AABB> &groups,
    std::vector<Light> &lights,
    int depth);