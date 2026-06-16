#include "geometric.cuh"
#include <cmath>

std::ostream &operator<<(std::ostream &os, const float3 &dir){
    os << dir.x << " " << dir.y << " " << dir.z;
    return os;
}
std::istream &operator>>(std::istream &is, float3 &dir){
    is >> dir.x >> dir.y >> dir.z;
    return is;
}

float3 to_cv3(const glm::vec3 &v){ return { v.x, v.y, v.z }; }

CudaMaterial_Old to_cmtl_old(const Material_Old &m){
    CudaMaterial_Old cm;
    cm.Kd = to_cv3(m.Kd);
    cm.Kg = to_cv3(m.Kg);
    cm.Ks = to_cv3(m.Ks);
    cm.refract = m.refract;
    cm.reflect = m.reflect;
    cm.glossy = m.glossy;
    cm.exp = m.exp;
    return cm;
}

float cpu_length(const float3 &v){
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

CudaMaterial to_cmtl(const Material &m){
    CudaMaterial cm;

    // 直接一對一映射 PBR 參數
    cm.base_color = make_float3(m.base_color.x, m.base_color.y, m.base_color.z);
    cm.roughness = m.roughness;
    cm.metallic = m.metallic;
    cm.eta = m.eta;

    // 依據物理特性判定材質類型 (對應我們前一次的 pbrt-v4 修改)
    if(cm.eta > 0.0f){
        cm.type = MaterialType::MAT_DIELECTRIC; // 玻璃 / 透射材質
    }
    else if(cm.metallic > 0.0f){
        cm.type = MaterialType::MAT_CONDUCTOR;  // 金屬 / 導體
    }
    else{
        cm.type = MaterialType::MAT_UBER;       // 一般 Diffuse / 粗糙表面
    }

    return cm;
}

float3 normalize_cuda(const float3 &v){
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return { v.x / len, v.y / len, v.z / len };
}

