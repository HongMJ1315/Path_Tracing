#include "object.h"
#include <algorithm>
#include <cmath>
#include <corecrt_math_defines.h>

bool Sphere::check_intersect(const Ray &ray, float &t, float &u, float &v,
    float tMin, float tMax) const{
    glm::vec3 D = glm::normalize(ray.vec);
    const glm::vec3 O = ray.point, C = center, S = scale;
    glm::vec3 OC = O - C;

    const float a = S.x * D.x * D.x + S.y * D.y * D.y + S.z * D.z * D.z;
    const float b = 2.0f * (S.x * D.x * OC.x + S.y * D.y * OC.y + S.z * D.z * OC.z);
    const float c = S.x * OC.x * OC.x + S.y * OC.y * OC.y + S.z * OC.z * OC.z - r * r;

    const float disc = b * b - 4.0f * a * c;
    if(disc < 0.0f) return false;

    const float sdisc = std::sqrt(std::max(0.0f, disc));
    float t0 = (-b - sdisc) / (2.0f * a);
    float t1 = (-b + sdisc) / (2.0f * a);
    if(t0 > t1) std::swap(t0, t1);

    float tCand = (t0 >= tMin) ? t0 : t1;
    if(tCand < tMin || tCand > tMax) return false;

    t = tCand;

    glm::vec3 P = O + glm::normalize(D) * t;
    glm::vec3 n_local = glm::normalize(glm::vec3(
        S.x * (P.x - C.x),
        S.y * (P.y - C.y),
        S.z * (P.z - C.z)
    ));

    float theta = std::atan2(n_local.z, n_local.x); // [-pi, pi]
    if(theta < 0) theta += 2.0f * float(M_PI);
    u = theta / (2.0f * float(M_PI));  // [0,1)
    v = std::acos(glm::clamp(n_local.y, -1.0f, 1.0f)) / float(M_PI); // [0,1]
    return true;
}

glm::vec3 Sphere::normal_at(const glm::vec3 &P, const Ray &ray, float, float) const{
    glm::vec3 n = glm::normalize(glm::vec3(
        scale.x * (P.x - center.x),
        scale.y * (P.y - center.y),
        scale.z * (P.z - center.z)
    ));
    if(glm::dot(n, ray.vec) > 0.0f) n = -n; // 面向調整
    return n;
}

glm::vec3 Triangle::geom_normal() const{
    return glm::normalize(glm::cross(vert[1] - vert[0], vert[2] - vert[0]));
}

bool Triangle::check_intersect(const Ray &ray, float &t, float &u, float &v,
    float tMin, float tMax) const{
    const glm::vec3 &v0 = vert[0], &v1 = vert[1], &v2 = vert[2];
    const glm::vec3 e1 = v1 - v0, e2 = v2 - v0;

    const glm::vec3 pvec = glm::cross(ray.vec, e2);
    const float det = glm::dot(e1, pvec);
    if(std::fabs(det) < ESP) return false;

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray.point - v0;

    u = glm::dot(tvec, pvec) * invDet;
    if(u < 0.0f || u > 1.0f) return false;

    const glm::vec3 qvec = glm::cross(tvec, e1);
    v = glm::dot(ray.vec, qvec) * invDet;
    if(v < 0.0f || (u + v) > 1.0f) return false;

    t = glm::dot(e2, qvec) * invDet;
    if(t < tMin || t > tMax) return false;

    return true;
}

glm::vec3 Triangle::normal_at(const glm::vec3 &, const Ray &ray,
    float, float) const{
    glm::vec3 n = geom_normal();
    if(glm::dot(n, ray.vec) > 0.0f) n = -n;
    return n;
}
