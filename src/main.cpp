// #include "GLinclude.h"
#include "ray.h"
#include "glsl.h"
#include "vao.h"
#include "mouse.h"
#include "mesh.h"
#include "light.h"
#include "object.h"
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <corecrt_math_defines.h>
#include <opencv2/opencv.hpp>


// Window size
glm::vec3 eye;
std::pair<int, int> resolution;
struct Camera{
    glm::vec3 ul, ur, ll, lr;
};




struct Hit{
    bool hit = false;
    float t = std::numeric_limits<float>::infinity();
    glm::vec3 pos{ 0.0f };
    glm::vec3 normal{ 0.0f };
    glm::vec3 bary{ 0.0f };   // Triangle: (w,u,v) ; Sphere: (u,v,0)
    float u = 0.0f, v = 0.0f;
    const Object *obj = nullptr;
    ObjKind kind = ObjKind::Unknown;
};


std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
}


inline Hit firstHit(const Ray &inRay, const std::vector<const Object *> &scene,
    float tMin = 1e-4f, float tMax = std::numeric_limits<float>::infinity()){
    Ray ray = inRay;
    ray.vec = glm::normalize(ray.vec);

    Hit best;
    for(const Object *obj : scene){
        float t, u, v;
        if(!obj->check_intersect(ray, t, u, v, tMin, tMax)) continue;
        if(t < best.t){
            best.hit = true;
            best.t = t;
            best.pos = ray.point + t * ray.vec;
            best.u = u;
            best.v = v;
            best.obj = obj;

            if(dynamic_cast<const Sphere *>(obj))   best.kind = ObjKind::Sphere;
            else if(dynamic_cast<const Triangle *>(obj)) best.kind = ObjKind::Triangle;
            else best.kind = ObjKind::Unknown;

            // 法向
            best.normal = obj->normal_at(best.pos, ray, u, v);

            // 重心 or UV
            if(best.kind == ObjKind::Triangle){
                best.bary = glm::vec3(1.0f - u - v, u, v); // (w,u,v)
            }
            else if(best.kind == ObjKind::Sphere){
                best.bary = glm::vec3(u, v, 0.0f);
            }
        }
    }
    return best;
}


glm::vec3 path_tracing(Ray ray,
    std::vector<Sphere> &balls,
    std::vector<Triangle> &triangles){
    std::vector<const Object *> scene;
    scene.reserve(balls.size() + triangles.size());
    for(const auto &s : balls)      scene.push_back(&s);
    for(const auto &t : triangles)  scene.push_back(&t);

    Hit h = firstHit(ray, scene);
    if(!h.hit) return glm::vec3(0.0f);
    if(h.kind == ObjKind::Sphere){
        // std::cout << h.pos << std::endl;
        // std::cout << ray.point + ray.vec << std::endl;
        // std::cout << "====" << std::endl;
    }

    glm::vec3 n = glm::normalize(h.normal);
    glm::vec3 color = glm::vec3(255 * (h.pos.x + 1.0) / 2.0,
        255 * (h.pos.y + 1.0) / 2.0,
        255 * (h.pos.z + 1.0) / 2.0);
    // std::cout << h.pos << std::endl;
    return color;
}

int main(int argc, char **argv){
    (void) argc; (void) argv;

    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(1920, 1080, "OBJ + Tetra with Vertex Colors", nullptr, nullptr);
    if(!window){
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if(!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    // Callbacks
    // glfwSetFramebufferSizeCallback(window, reshape);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // GL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);


    /*--------------------------
    input
    --------------------------*/
    std::fstream input("../../input.txt");
    std::vector<Sphere> balls;
    std::vector<Triangle> triangles;
    Camera camera;
    char t;
    Material mtl;
    while(input >> t){
        if(t == 'E'){
            float x, y, z;
            input >> eye;
        }
        else if(t == 'O'){
            input >> camera.ul;
            input >> camera.ur;
            input >> camera.ll;
            input >> camera.lr;
        }
        else if(t == 'R'){
            input >> resolution.first >> resolution.second;
        }
        else if(t == 'S'){
            Sphere s;
            input >> s.center >> s.r;
            s.scale = glm::vec3(1, 1, 1);
            // s.mtl = mtl;
            balls.push_back(s);
        }
        else if(t == 'T'){
            Triangle tri;
            glm::vec3 vert;
            for(int i = 0; i < 3; i++){
                input >> vert;
                tri.vert[i] = vert;
            }
            // tri.mtl = mtl;
            triangles.push_back(tri);
        }
        else if(t == 'M'){
            float ka, kd, ks;
            glm::vec3 color;
            std::cin >> color;
            std::cin >> ka >> kd >> ks >> mtl.exp >> mtl.reflect;
            mtl.Ka = color * ka;
            mtl.Kd = color * kd;
            mtl.Ks = color * ks;
        }
    }

    std::cout << "Eye Position: (" << eye.x << ", " << eye.y << ", " << eye.z << ")" << std::endl;
    std::cout << "Screen Info: " << std::endl;
    std::cout << "  UL: " << camera.ul << std::endl;
    std::cout << "  UR: " << camera.ur << std::endl;
    std::cout << "  LL: " << camera.ll << std::endl;
    std::cout << "  LR: " << camera.lr << std::endl;
    std::cout << "Ball:" << std::endl;
    for(auto b : balls){
        std::cout << "  Location: " << b.center << std::endl;
        std::cout << "  R: " << b.r << std::endl;
    }
    std::cout << "Trainagle:" << std::endl;
    for(auto t : triangles){
        std::cout << "  Vertex: (" << std::endl;
        for(int i = 0; i < 3; i++){
            std::cout << "    vertex " << i << ": " << t.vert[i] << std::endl;
        }
        std::cout << "  )" << std::endl;
    }

    glm::vec3 width = camera.lr - camera.ll,
        height = camera.ul - camera.ll;
    std::cout << width / (float) resolution.first << "\n" << height / (float) resolution.first << std::endl;
    glm::vec3 dx = width / (float) resolution.first;
    glm::vec3 dy = height / (float) resolution.second;
    cv::Mat img(resolution.first, resolution.second, CV_8UC3);  // 三通道 (BGR)
    for(int i = 0; i < resolution.first; i++){
        for(int j = 0; j < resolution.second; j++){
            glm::vec3 pixel_pos = camera.ll + dx * (float) i + dy * (float) j;
            // std::cout << pixel_pos << std::endl;
            glm::vec3 dir = pixel_pos - eye;
            Ray ray(eye, dir);
            glm::vec3 pixel_color = path_tracing(ray, balls, triangles);
            cv::Vec3b &pixel = img.at<cv::Vec3b>(j, i);
            pixel[2] = static_cast<uchar>(pixel_color.x);
            pixel[0] = static_cast<uchar>(pixel_color.y);
            pixel[1] = static_cast<uchar>(pixel_color.z);
            // std::cout << pixel_pos << std::endl;
        }
    }
    cv::Mat filp_img(resolution.first, resolution.second, CV_8UC3);  // 三通道 (BGR)
    cv::flip(img, filp_img, 0);
    if(!cv::imwrite("color_output.png", filp_img)){
        std::cerr << "彩色圖片輸出失敗！" << std::endl;
    }




    while(!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glfwTerminate();
    return 0;
}
