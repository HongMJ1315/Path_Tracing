#include "GLinclude.h"
#include "glsl.h"
#include "rt.h"
#include "texture.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <corecrt_math_defines.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <omp.h>

#define RENDER_THREADS 5

// Window size
glm::vec3 eye;
std::pair<int, int> resolution;
struct Camera{
    // glm::vec3 ul, ur, ll, lr;
    glm::vec3 eye; //eye position
    glm::vec3 look_at; // look position
    glm::vec3 view_up; //view up vector
    float fov;
};

std::ostream &operator<<(std::ostream &os, const glm::vec3 vec){
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::istream &operator>>(std::istream &is, glm::vec3 &vec){
    is >> vec.x >> vec.y >> vec.z;
    return is;
}




int render_array[500 * 500] = {};


int main(int argc, char **argv){
    (void) argc; (void) argv;
    omp_set_num_threads(RENDER_THREADS);  // 在 main() 一開始設

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

    GLFWwindow *window = glfwCreateWindow(256, 256, "OBJ + Tetra with Vertex Colors", nullptr, nullptr);
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

    // GL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(1.f, 1.f, 1.0f, 1.0f);

    /*--------------------------
    input
    --------------------------*/
    std::fstream input("../../input.txt");
    std::vector<Sphere> balls;
    std::vector<Triangle> triangles;
    std::vector<Light> lights;
    Camera camera;
    char t;
    Material mtl;
    while(input >> t){
        if(t == 'E'){
            float x, y, z;
            input >> camera.eye;
        }
        else if(t == 'V'){
            input >> camera.look_at >> camera.view_up;
        }
        else if(t == 'F'){
            input >> camera.fov;
            // std::cout << "read F" << std::endl;
        }
        else if(t == 'R'){
            input >> resolution.first >> resolution.second;
            // std::cout << "read R" << std::endl;
        }
        else if(t == 'S'){
            Sphere s;
            input >> s.center >> s.r;
            s.scale = glm::vec3(1, 1, 1);
            s.mtl = mtl;
            balls.push_back(s);
            // std::cout << "read S" << std::endl;
        }
        else if(t == 'T'){
            Triangle tri;
            glm::vec3 vert;
            for(int i = 0; i < 3; i++){
                input >> vert;
                tri.vert[i] = vert;
            }
            tri.mtl = mtl;
            triangles.push_back(tri);
            // std::cout << "read T" << std::endl;
        }
        else if(t == 'M'){
            float ka, kd, ks;
            glm::vec3 color;
            input >> color;
            input >> ka >> kd >> ks >> mtl.exp >> mtl.reflect;
            mtl.Ka = color * ka;
            mtl.Kd = color * kd;
            mtl.Ks = color * ks;
            // std::cout << color << " " << mtl.Ka << " " << mtl.Kd << " " << mtl.Ks << std::endl;
            // std::cout << "read M" << std::endl;
        }
        else if(t == 'L'){
            Light light;
            input >> light.dir;
            lights.push_back(light);
        }
    }

    const int W = resolution.first;   // width
    const int H = resolution.second;  // height
    const float aspect = float(W) / float(H);

    glViewport(0, 0, 256, 256);
    glDisable(GL_DEPTH_TEST);

    gen_texture(resolution.first, resolution.second);

    GLuint vao = get_vao(), prog = get_shader(), tex = get_texture();

    std::cout << "VAO: " << vao << std::endl;
    std::cout << "program: " << prog << std::endl;
    std::cout << "Eye Position: " << camera.eye << std::endl;
    std::cout << "Screen Info: " << std::endl;
    std::cout << "  Look At: " << camera.look_at << std::endl;
    std::cout << "  View Up: " << camera.view_up << std::endl;
    std::cout << "  FOV: " << camera.fov << std::endl;
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



    auto deg2rad = [](float d){ return d * float(M_PI) / 180.0f; };
    const float fov_rad = deg2rad(camera.fov);
    const glm::vec3 cam_eye = camera.eye;
    glm::vec3 f = glm::normalize(camera.look_at - camera.eye); // forward
    glm::vec3 up = glm::normalize(camera.view_up);
    glm::vec3 r = glm::normalize(glm::cross(f, up));           // right
    glm::vec3 u = glm::cross(r, f);                            // true up

    // ---- 影像平面四角（dist=1） ----
    const float dist = 1.0f;
    const glm::vec3 center = cam_eye + dist * f;
    const float half_h = std::tan(0.5f * fov_rad); // vertical fov
    const float half_w = half_h * aspect;

    glm::vec3 UL = center + u * half_h - r * half_w;
    glm::vec3 UR = center + u * half_h + r * half_w;
    glm::vec3 LL = center - u * half_h - r * half_w;
    glm::vec3 LR = center - u * half_h + r * half_w;

    glm::vec3 dx = (UR - UL) / float(W);
    glm::vec3 dy = (LL - UL) / float(H);


    std::vector<const Object *> scene;
    scene.reserve(balls.size() + triangles.size());
    for(const auto &s : balls)      scene.push_back(&s);
    for(const auto &t : triangles)  scene.push_back(&t);

    /*
    for(int j = 0; j < H; ++j){
        for(int i = 0; i < W; ++i){
            std::cout << "(" << i << ", " << j << ")" << std::endl;
            // 像素中心
            // std::cout << "==========" << std::endl;
            glm::vec3 col(0.0f);
            for(int k = 0; k < SAMPLE; k++){
                glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f) + dy * (float(j) + 0.5f);
                glm::vec3 dir = glm::normalize(pixel_pos - cam_eye);
                glm::vec3 esp = glm::vec3(get_esp(), get_esp(), get_esp());
                // std::cout << esp << std::endl;
                dir += esp;
                dir = glm::normalize(dir);
                // std::cout << dir << std::endl;

                Ray ray(cam_eye, dir);

                col += path_tracing(ray, scene, lights, 0);
            }
            col = (col / SAMPLE) * 255.f;
            // std::cout << col << std::endl;
            // Clamp & BGR
            col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(255.0f));
            cv::Vec3b &pix = img.at<cv::Vec3b>(j, i);
            pix[0] = (uchar) col.z; // B
            pix[1] = (uchar) col.y; // G
            pix[2] = (uchar) col.x; // R
        }
    }

    if(!cv::imwrite("color_output.png", img)){
        std::cerr << "彩色圖片輸出失敗！" << std::endl;
    }
    else std::cerr << "Rendering Finish" << std::endl;
*/

    std::vector<unsigned char> framebuffer(W * H * 3, 0);

    int pixel_cursor = 0;
    const int k_pixels_per_frame = 200;

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    glUseProgram(0);

    bool is_writed = false;

    for(int i = 0; i < W * H; i++){
        render_array[i] = i;
    }

    std::random_device rd;
    unsigned int seed = rd();
    std::mt19937 mt_rand = std::mt19937(seed);
    std::shuffle(render_array, render_array + W * H, mt_rand);

    auto start_time = std::chrono::steady_clock::now();

    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // /*
        int end = std::min(W * H, pixel_cursor + k_pixels_per_frame);

#pragma omp parallel for schedule(dynamic, 128)
        for(int p = pixel_cursor; p < end; ++p){
            int j = render_array[p] / W;              // row
            int i = render_array[p] % W;              // col

            glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f) + dy * (float(j) + 0.5f);
            glm::vec3 dir = glm::normalize(pixel_pos - cam_eye);

            Ray ray(cam_eye, dir);

            glm::vec3 col(0.0f);

            for(int k = 0; k < SAMPLE; k++){
                // auto get_esp = [](){
                //     std::random_device rd;
                //     unsigned int seed = rd();
                //     std::mt19937 mt_rand = std::mt19937(seed);
                //     return (mt_rand() * 10000 % 10) / 5000.f;
                // };

                glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f) + dy * (float(j) + 0.5f);
                glm::vec3 dir = glm::normalize(pixel_pos - cam_eye);
                glm::vec3 esp = glm::vec3(get_esp(), get_esp(), get_esp());
                // std::cout << esp << std::endl;
                dir += esp;
                dir = glm::normalize(dir);
                // std::cout << dir << std::endl;

                Ray ray(cam_eye, dir);

                col += path_tracing(ray, scene, lights, 0);
            }

            col = col / SAMPLE;

            col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f));
            col = glm::pow(col, glm::vec3(1.0f / 2.2f)); // gamma

            size_t idx = (size_t(H - 1 - j) * W + i) * 3;
            framebuffer[idx + 0] = (unsigned char) (col.r * 255.0f);
            framebuffer[idx + 1] = (unsigned char) (col.g * 255.0f);
            framebuffer[idx + 2] = (unsigned char) (col.b * 255.0f);
        }
        // std::cout << "update" << std::endl;
        pixel_cursor = end;

        // */
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, framebuffer.data());


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glUseProgram(0);

        glfwSwapBuffers(window);
        // std::cout << "there" << std::endl;

        if(pixel_cursor >= W * H && !is_writed){
            auto end_time = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Elapsed time: " << diff.count() << " ms" << std::endl;
            std::cout << "Render Finish" << std::endl;
            is_writed = true;
            cv::Mat img_rgb(H, W, CV_8UC3, framebuffer.data());
            cv::Mat img_bgr, img_bgr_Flip;

            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            cv::flip(img_bgr, img_bgr_Flip, 0);

            if(!cv::imwrite("color_output.png", img_bgr_Flip)){
                std::cerr << "彩色圖片輸出失敗！" << std::endl;
            }
        }
    }


    glfwTerminate();
    return 0;
}
