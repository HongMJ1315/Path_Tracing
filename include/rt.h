#include "object.h"
#include "rng.h"
#include "bdpt.cuh"
#include <vector>
#include <cstring> // for memcpy
#include <vector>
#include <map>

#define MAX_DPETH 1
#define SAMPLE 5.f

#define LIGHT_DEPTH 1
#define TRACE_MODE 0

#define EYE_DEPTH 3

struct LightVertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 wi;
    glm::vec3 throughput;
    Object *obj;
};

struct EyeVertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 wi;
    glm::vec3 throughput;
    Object *obj;
};


struct EyeRayInfo{
    int i, j, sample;
    Ray ray;
};

std::ostream &operator<<(std::ostream &os, const glm::vec3 vec);
std::istream &operator>>(std::istream &is, glm::vec3 &vec);

std::vector<glm::vec3> run_cuda_eye_light_connect(
    int W, int H,
    std::map<int, AABB> &groups
);

inline Hit first_hit(Ray &inRay, std::map<int, AABB> &groups,
    float tMin, float tMax);

void init_lightray(std::map<int, AABB> &groups);

glm::vec3 sample_hemisphere_cosine(const glm::vec3 &normal);

glm::vec3 lightray_tracer(Ray light_ray, std::map<int, AABB> &groups, glm::vec3 throughput);

void init_light_group();

void resize_screen_info(int W, int H);

void init_eyeray(std::map<int, AABB> &groups, std::vector<EyeRayInfo> &eyeray,
    int W, int H);

std::vector<EyeVertex> eyeray_tracer(Ray &ray,
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