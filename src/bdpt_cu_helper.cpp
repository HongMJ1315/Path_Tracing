#include "bdpt_cu_helper.h"

std::vector<CudaSphere> cuda_spheres;
std::vector<CudaTriangle> cuda_triangles;
std::vector<CudaLight> cuda_lights;
CudaVec3 scene_max_bound = { -1e9f, -1e9f, -1e9f };
CudaVec3 scene_min_bound = { 1e9f, 1e9f, 1e9f };
int light_sample = 0;
// 輔助函式：將 glm::vec3 轉為 CudaVec3
CudaVec3 to_cv3(const glm::vec3 &v){
    return { v.x, v.y, v.z };
}

// 輔助函式：將 Material 轉為 CudaMaterial
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


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
}

void move_data_to_cuda(std::map<int, AABB> groups, std::vector<CudaLight> &lights, int light_sample){
    /*--------------------------
    Move to CUDA Formate
    --------------------------*/
    for(auto &g : groups){
        AABB *box = &(g.second);
        for(auto &obj : box->objs){
            const Sphere *sph = dynamic_cast<const Sphere *>(obj);
            const Triangle *tri = dynamic_cast<const Triangle *>(obj);
            if(sph){
                CudaSphere csph;
                csph.center = to_cv3(sph->center);
                csph.r = sph->r;
                csph.mtl = to_cmtl(sph->mtl);
                csph.id = sph->obj_id;
                cuda_spheres.push_back(csph);
                scene_max_bound.x = std::max(scene_max_bound.x, sph->center.x + sph->r);
                scene_max_bound.y = std::max(scene_max_bound.y, sph->center.y + sph->r);
                scene_max_bound.z = std::max(scene_max_bound.z, sph->center.z + sph->r);
                scene_min_bound.x = std::min(scene_min_bound.x, sph->center.x - sph->r);
                scene_min_bound.y = std::min(scene_min_bound.y, sph->center.y - sph->r);
                scene_min_bound.z = std::min(scene_min_bound.z, sph->center.z - sph->r);
            }
            else if(tri){
                CudaTriangle ctri;
                ctri.v0 = to_cv3(tri->vert[0]);
                ctri.v1 = to_cv3(tri->vert[1]);
                ctri.v2 = to_cv3(tri->vert[2]);
                ctri.mtl = to_cmtl(tri->mtl);
                ctri.id = tri->obj_id;
                cuda_triangles.push_back(ctri);
                scene_max_bound.x = std::max({ scene_max_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                scene_max_bound.y = std::max({ scene_max_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                scene_max_bound.z = std::max({ scene_max_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
                scene_min_bound.x = std::min({ scene_min_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                scene_min_bound.y = std::min({ scene_min_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                scene_min_bound.z = std::min({ scene_min_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
            }
        }
    }

    for(auto &l : lights){
        l.dir = normalize_cuda(l.dir);
        l.illum.x /= (float) light_sample;
        l.illum.y /= (float) light_sample;
        l.illum.z /= (float) light_sample;
        cuda_lights.push_back(l);
    }

    ::light_sample = light_sample;

}

void run_cuda_bdpt(CudaCamera cam, CudaVec3 *image_buffer, int light_depth, int eye_depth, int W, int H){
    /*--------------------------
    Call CUDA BDPT Kernel
    --------------------------*/
    bdpt_render_wrapper(
        cuda_lights.data(), cuda_lights.size(),
        cuda_spheres.data(), cuda_spheres.size(),
        cuda_triangles.data(), cuda_triangles.size(),
        scene_min_bound, scene_max_bound,
        cam, image_buffer, W, H, light_depth, light_sample, eye_depth
    );
}