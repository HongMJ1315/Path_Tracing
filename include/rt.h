#include "object.h"
#include "rng.h"
#include <vector>
#include <map>

#define MAX_DPETH 3
#define SAMPLE 25.f

std::ostream &operator<<(std::ostream &os, const glm::vec3 vec);
std::istream &operator>>(std::istream &is, glm::vec3 &vec);

inline Hit first_hit(const Ray &inRay, const std::map<int, AABB> &groups,
    float tMin, float tMax);

glm::vec3 phong(const Hit &hit, const std::vector<Light> &lights,
    const std::map<int, AABB> &groups);


glm::vec3 path_tracing(Ray ray,
    const std::map<int, AABB> &groups,
    std::vector<Light> &lights,
    int depth);