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

inline float random_float(float a, float b){
    // C++14: thread_local 需要分離初始化
    thread_local std::mt19937 rng(
        []{
        std::random_device rd;
        unsigned int tid = static_cast<unsigned int>(omp_get_thread_num());
        return std::mt19937::result_type(
            rd() ^ (0x9E3779B9u + tid + (tid << 6) + (tid >> 2))
        );
    }()
        );

    // 若 a > b 則交換，避免不合法的分佈參數
    if(a > b) std::swap(a, b);

    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

inline glm::vec3 sample_hemisphere_uniform(const glm::vec3 &N){
    float u1 = random_float(0.0f, 1.0f);
    float u2 = random_float(0.0f, 1.0f);

    float cosTheta = u1;                              // cosθ ~ U[0,1]
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * 3.14159265358979323846f * u2;

    float x = sinTheta * std::cos(phi);
    float z = sinTheta * std::sin(phi);
    float y = cosTheta;                               // 半球朝 +Y

    glm::vec3 local_dir(x, y, z);                    // 以 +Y 為法向的半球

    glm::vec3 n = glm::normalize(N);

    glm::vec3 helper = (std::fabs(n.y) < 0.999f)
        ? glm::vec3(0.0f, 1.0f, 0.0f)
        : glm::vec3(1.0f, 0.0f, 0.0f);

    glm::vec3 tangent = glm::normalize(glm::cross(helper, n));
    glm::vec3 bitangent = glm::cross(n, tangent);

    glm::vec3 world_dir =
        local_dir.x * tangent +
        local_dir.y * n +
        local_dir.z * bitangent;

    return glm::normalize(world_dir); // 理論上長度為 1，normalize 做保險
}



inline float get_rand(){
    return (std::abs(int(rng_uniform01() * 10000) % 100)) / 100.f;
}                    // [0,1)
inline float get_esp(){
    return ((int) (rng_uniform01() * 10000) % 10) / 5000.f;
}
