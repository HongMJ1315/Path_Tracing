#include "object.h"
#include "rng.h"
#include <vector>

#define MAX_DPETH 10
#define SAMPLE 25.f


inline Hit first_hit(const Ray &inRay, const std::vector<const Object *> &scene,
    float tMin, float tMax);

glm::vec3 phong(const Hit &hit, const std::vector<Light> &lights,
    const std::vector<const Object *> &scene);


glm::vec3 path_tracing(Ray ray,
    std::vector<const Object *> scene,
    std::vector<Light> &lights,
    int depth);