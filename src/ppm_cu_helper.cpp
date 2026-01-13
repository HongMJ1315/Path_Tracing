#include "ppm_cu_helper.h"


namespace ppm_ns{
    std::vector<CudaSphere> cuda_spheres;
    std::vector<CudaTriangle> cuda_triangles;
    std::vector<CudaLight> cuda_lights;
    CudaVec3 scene_max_bound = { -1e9f, -1e9f, -1e9f };
    CudaVec3 scene_min_bound = { 1e9f, 1e9f, 1e9f };
    int light_sample = 0;
}


void move_data_to_cuda_ppm(std::map<int, AABB> groups, std::vector<CudaLight> &lights, int light_sample){
    ppm_ns::cuda_spheres.clear(); ppm_ns::cuda_triangles.clear(); ppm_ns::cuda_lights.clear();
    for(auto &g : groups){
        AABB *box = &(g.second);
        for(auto &obj : box->objs){
            const Sphere *sph = dynamic_cast<const Sphere *>(obj);
            const Triangle *tri = dynamic_cast<const Triangle *>(obj);
            if(sph){
                CudaSphere csph; csph.center = to_cv3(sph->center); csph.r = sph->r; csph.mtl = to_cmtl(sph->mtl); csph.id = sph->obj_id;
                ppm_ns::cuda_spheres.push_back(csph);
                ppm_ns::scene_max_bound.x = std::max(ppm_ns::scene_max_bound.x, sph->center.x + sph->r);
                ppm_ns::scene_max_bound.y = std::max(ppm_ns::scene_max_bound.y, sph->center.y + sph->r);
                ppm_ns::scene_max_bound.z = std::max(ppm_ns::scene_max_bound.z, sph->center.z + sph->r);
                ppm_ns::scene_min_bound.x = std::min(ppm_ns::scene_min_bound.x, sph->center.x - sph->r);
                ppm_ns::scene_min_bound.y = std::min(ppm_ns::scene_min_bound.y, sph->center.y - sph->r);
                ppm_ns::scene_min_bound.z = std::min(ppm_ns::scene_min_bound.z, sph->center.z - sph->r);
            }
            else if(tri){
                CudaTriangle ctri; ctri.v0 = to_cv3(tri->vert[0]); ctri.v1 = to_cv3(tri->vert[1]); ctri.v2 = to_cv3(tri->vert[2]); ctri.mtl = to_cmtl(tri->mtl); ctri.id = tri->obj_id;
                ppm_ns::cuda_triangles.push_back(ctri);
                ppm_ns::scene_max_bound.x = std::max({ ppm_ns::scene_max_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                ppm_ns::scene_max_bound.y = std::max({ ppm_ns::scene_max_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                ppm_ns::scene_max_bound.z = std::max({ ppm_ns::scene_max_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
                ppm_ns::scene_min_bound.x = std::min({ ppm_ns::scene_min_bound.x, tri->vert[0].x, tri->vert[1].x, tri->vert[2].x });
                ppm_ns::scene_min_bound.y = std::min({ ppm_ns::scene_min_bound.y, tri->vert[0].y, tri->vert[1].y, tri->vert[2].y });
                ppm_ns::scene_min_bound.z = std::min({ ppm_ns::scene_min_bound.z, tri->vert[0].z, tri->vert[1].z, tri->vert[2].z });
            }
        }
    }
    for(auto &l : lights){
        l.dir = normalize_cuda(l.dir);
        // l.illum /= light_sample; // Flux 不需要除以採樣數，在 Kernel 算 Power 時會處理
        ppm_ns::cuda_lights.push_back(l);
    }
    ppm_ns::light_sample = light_sample;
}

void run_cuda_ppm(CudaCamera cam, CudaVec3 *image_buffer, int light_depth, int eye_depth, int W, int H){
    ppm_render_wrapper(
        ppm_ns::cuda_lights.data(), ppm_ns::cuda_lights.size(),
        ppm_ns::cuda_spheres.data(), ppm_ns::cuda_spheres.size(),
        ppm_ns::cuda_triangles.data(), ppm_ns::cuda_triangles.size(),
        ppm_ns::scene_min_bound, ppm_ns::scene_max_bound,
        cam, image_buffer, W, H, light_depth, ppm_ns::light_sample, eye_depth
    );
}