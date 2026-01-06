#include "object.h"
#include "bdpt_cu.cuh"
#include <iostream>
#include <map>


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec);
std::istream &operator>>(std::istream &is, glm::vec3 &vec);
void move_data_to_cuda(std::map<int, AABB> groups, std::vector<CudaLight> &cuda_lights, int);
void run_cuda_bdpt(CudaCamera cam, CudaVec3 *image_buffer,int light_depth, int eye_depth, int W, int H);
