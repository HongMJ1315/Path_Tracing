#include "pt_cu_helper.h"

namespace pt_ns{
    std::vector<CudaSphere> cuda_spheres;
    std::vector<CudaTriangle> cuda_triangles;
    std::vector<CudaLight> cuda_lights;
    float3 scene_max_bound = { -1e9f, -1e9f, -1e9f };
    float3 scene_min_bound = { 1e9f, 1e9f, 1e9f };
    int light_sample = 0;
}
void move_data_to_cuda_pt(std::map<int, AABB> groups, std::vector<CudaLight> &lights, int light_sample){
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
                pt_ns::cuda_spheres.push_back(csph);
                pt_ns::scene_max_bound.x = std::max(pt_ns::scene_max_bound.x, sph->center.x + sph->r);
                pt_ns::scene_max_bound.y = std::max(pt_ns::scene_max_bound.y, sph->center.y + sph->r);
                pt_ns::scene_max_bound.z = std::max(pt_ns::scene_max_bound.z, sph->center.z + sph->r);
                pt_ns::scene_min_bound.x = std::min(pt_ns::scene_min_bound.x, sph->center.x - sph->r);
                pt_ns::scene_min_bound.y = std::min(pt_ns::scene_min_bound.y, sph->center.y - sph->r);
                pt_ns::scene_min_bound.z = std::min(pt_ns::scene_min_bound.z, sph->center.z - sph->r);
            }
            else if(tri){
                CudaTriangle ctri;
                ctri.v0 = to_cv3(tri->vert[0]);
                ctri.v1 = to_cv3(tri->vert[1]);
                ctri.v2 = to_cv3(tri->vert[2]);
                ctri.mtl = to_cmtl(tri->mtl);
                ctri.id = tri->obj_id;
                pt_ns::cuda_triangles.push_back(ctri);
                pt_ns::scene_max_bound.x = std::max({ pt_ns::scene_max_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                pt_ns::scene_max_bound.y = std::max({ pt_ns::scene_max_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                pt_ns::scene_max_bound.z = std::max({ pt_ns::scene_max_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
                pt_ns::scene_min_bound.x = std::min({ pt_ns::scene_min_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                pt_ns::scene_min_bound.y = std::min({ pt_ns::scene_min_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                pt_ns::scene_min_bound.z = std::min({ pt_ns::scene_min_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
            }
        }
    }

    for(auto l : lights){
        l.dir = normalize_cuda(l.dir);
        l.illum.x /= (float) light_sample;
        l.illum.y /= (float) light_sample;
        l.illum.z /= (float) light_sample;
        pt_ns::cuda_lights.push_back(l);
    }

    pt_ns::light_sample = light_sample;

}

void run_cuda_pt(CudaCamera cam, float3 *image_buffer, int light_depth, int eye_depth, int W, int H){
    /*--------------------------
    Call CUDA PT Kernel
    --------------------------*/
    pt_render_wrapper(
        pt_ns::cuda_lights.data(), pt_ns::cuda_lights.size(),
        pt_ns::cuda_spheres.data(), pt_ns::cuda_spheres.size(),
        pt_ns::cuda_triangles.data(), pt_ns::cuda_triangles.size(),
        pt_ns::scene_min_bound, pt_ns::scene_max_bound,
        cam, image_buffer, W, H, light_depth, pt_ns::light_sample, eye_depth
    );
}