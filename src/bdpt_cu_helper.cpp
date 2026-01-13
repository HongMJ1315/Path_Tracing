#include "bdpt_cu_helper.h"

namespace bdpt_ns{
    std::vector<CudaSphere> cuda_spheres;
    std::vector<CudaTriangle> cuda_triangles;
    std::vector<CudaLight> cuda_lights;
    CudaVec3 scene_max_bound = { -1e9f, -1e9f, -1e9f };
    CudaVec3 scene_min_bound = { 1e9f, 1e9f, 1e9f };
    int light_sample = 0;
}
void move_data_to_cuda_bdpt(std::map<int, AABB> groups, std::vector<CudaLight> &lights, int light_sample){
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
                bdpt_ns::cuda_spheres.push_back(csph);
                bdpt_ns::scene_max_bound.x = std::max(bdpt_ns::scene_max_bound.x, sph->center.x + sph->r);
                bdpt_ns::scene_max_bound.y = std::max(bdpt_ns::scene_max_bound.y, sph->center.y + sph->r);
                bdpt_ns::scene_max_bound.z = std::max(bdpt_ns::scene_max_bound.z, sph->center.z + sph->r);
                bdpt_ns::scene_min_bound.x = std::min(bdpt_ns::scene_min_bound.x, sph->center.x - sph->r);
                bdpt_ns::scene_min_bound.y = std::min(bdpt_ns::scene_min_bound.y, sph->center.y - sph->r);
                bdpt_ns::scene_min_bound.z = std::min(bdpt_ns::scene_min_bound.z, sph->center.z - sph->r);
            }
            else if(tri){
                CudaTriangle ctri;
                ctri.v0 = to_cv3(tri->vert[0]);
                ctri.v1 = to_cv3(tri->vert[1]);
                ctri.v2 = to_cv3(tri->vert[2]);
                ctri.mtl = to_cmtl(tri->mtl);
                ctri.id = tri->obj_id;
                bdpt_ns::cuda_triangles.push_back(ctri);
                bdpt_ns::scene_max_bound.x = std::max({ bdpt_ns::scene_max_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                bdpt_ns::scene_max_bound.y = std::max({ bdpt_ns::scene_max_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                bdpt_ns::scene_max_bound.z = std::max({ bdpt_ns::scene_max_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
                bdpt_ns::scene_min_bound.x = std::min({ bdpt_ns::scene_min_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                bdpt_ns::scene_min_bound.y = std::min({ bdpt_ns::scene_min_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                bdpt_ns::scene_min_bound.z = std::min({ bdpt_ns::scene_min_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
            }
        }
    }

    for(auto &l : lights){
        l.dir = normalize_cuda(l.dir);
        l.illum.x /= (float) light_sample;
        l.illum.y /= (float) light_sample;
        l.illum.z /= (float) light_sample;
        bdpt_ns::cuda_lights.push_back(l);
    }

    bdpt_ns::light_sample = light_sample;

}

void run_cuda_bdpt(CudaCamera cam, CudaVec3 *image_buffer, int light_depth, int eye_depth, int W, int H){
    /*--------------------------
    Call CUDA BDPT Kernel
    --------------------------*/
    bdpt_render_wrapper(
        bdpt_ns::cuda_lights.data(), bdpt_ns::cuda_lights.size(),
        bdpt_ns::cuda_spheres.data(), bdpt_ns::cuda_spheres.size(),
        bdpt_ns::cuda_triangles.data(), bdpt_ns::cuda_triangles.size(),
        bdpt_ns::scene_min_bound, bdpt_ns::scene_max_bound,
        cam, image_buffer, W, H, light_depth, bdpt_ns::light_sample, eye_depth
    );
}