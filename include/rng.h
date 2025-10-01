#include <random>
#include <omp.h>

inline float rng_uniform01(){
    thread_local std::mt19937 rng([]{
        std::random_device rd;
        unsigned int tid = static_cast<unsigned int>(omp_get_thread_num());
        return std::mt19937::result_type(rd() ^ (0x9E3779B9u + tid + (tid << 6) + (tid >> 2)));
    }());
    thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

inline float get_rand(){
    return (std::abs(int(rng_uniform01() * 10000) % 100)) / 100.f;
}                    // [0,1)
inline float get_esp(){
    return ((int) (rng_uniform01() * 10000) % 10) / 5000.f;
}
