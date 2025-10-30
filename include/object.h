#include <limits>
#include <glm/glm.hpp>
#include <vector>
#include "ray.h"

#define ESP 1e-6

// 前置宣告

enum class ObjKind{ Unknown, Sphere, Triangle };

struct Material{
    glm::vec3 Ka{ 0.0f };   // ambient
    glm::vec3 Kd{ 0.0f };   // diffuse
    glm::vec3 Ks{ 0.0f };   // specular
    float exp{ 32.0f };     // shininess
    float reflect{ 0.0f };  // reflectivity
    float refract{ 0.0f };  // reflectivity
};

struct Light{
    glm::vec3 dir;
    float illum = 1.0f;
};

class Object{
public:
    Material mtl;
    int obj_id;
    virtual ~Object() = default;

    virtual bool check_intersect(Ray &ray, float &t, float &u, float &v,
        float tMin = 1e-4f,
        float tMax = std::numeric_limits<float>::infinity()) = 0;

    virtual glm::vec3 normal_at(glm::vec3 &P,  Ray &ray,
        float u, float v) = 0;
};

class Sphere : public Object{
public:
    float r = 1.0f;
    glm::vec3 center{ 0.0f };
    glm::vec3 scale{ 1.0f, 1.0f, 1.0f };

    bool check_intersect(Ray &ray, float &t, float &u, float &v,
        float tMin = 1e-4f,
        float tMax = std::numeric_limits<float>::infinity())override;

    glm::vec3 normal_at(glm::vec3 &P, Ray &ray,
        float u, float v) override;
};

class Triangle : public Object{
public:
    glm::vec3 vert[3];

    glm::vec3 geom_normal();

    bool check_intersect(Ray &ray, float &t, float &u, float &v,
        float tMin = 1e-4f,
        float tMax = std::numeric_limits<float>::infinity()) override;

    glm::vec3 normal_at(glm::vec3 &P, Ray &ray,
        float u, float v) override;
};

struct Hit{
    bool hit = false;
    float t = std::numeric_limits<float>::infinity();
    glm::vec3 eye_dir{ 0.0f };
    glm::vec3 pos{ 0.0f };
    glm::vec3 normal{ 0.0f };
    glm::vec3 bary{ 0.0f };   // Triangle: (w,u,v) ; Sphere: (u,v,0)
    float u = 0.0f, v = 0.0f;
    const Object *obj = nullptr;
    ObjKind kind = ObjKind::Unknown;
};

class AABB{
public:
    glm::vec3 min = { 99999.f, 99999.f, 99999.f },
        max = { -99999.f, -99999.f, -99999.f };
    std::vector<Object *> objs;
    void add_obj(Object *obj);
    bool intersectAABB(Ray &r,
        float tMin = 1e-4, float tMax = std::numeric_limits<float>::infinity());
};