#include "bdpt_cu.cuh"
#include <map>


void move_data_to_cuda_bdpt(std::map<int, AABB> groups, std::vector<CudaLight> &cuda_lights, int);
void run_cuda_bdpt(CudaCamera cam, CudaVec3 *image_buffer, int light_depth, int eye_depth, int W, int H);
