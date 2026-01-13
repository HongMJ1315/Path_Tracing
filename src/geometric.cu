#include "geometric.cuh"
#include <cmath>

CudaVec3 to_cv3(const glm::vec3 &v){ return { v.x, v.y, v.z }; }

CudaMaterial to_cmtl(const Material &m){
    CudaMaterial cm;
    cm.Kd = to_cv3(m.Kd);
    cm.Kg = to_cv3(m.Kg);
    cm.Ks = to_cv3(m.Ks);
    cm.refract = m.refract;
    cm.reflect = m.reflect;
    cm.glossy = m.glossy;
    cm.exp = m.exp;
    return cm;
}

CudaVec3 normalize_cuda(const CudaVec3 &v){
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return { v.x / len, v.y / len, v.z / len };
}

