#include "GLinclude.h"
#include "object.h"
#include "glsl.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>
#include <corecrt_math_defines.h>
#include <opencv2/opencv.hpp>
#define MAX_DPETH 3
#define SAMPLE 25.f

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

std::mt19937 mt_rand;
float get_esp(){
    return (mt_rand() * 10000 % 10) / 5000.f;
}
float get_rand(){
    return (std::abs(int(mt_rand() * 10000) % 100)) / 100.f;
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
            best.eye_dir = glm::normalize(ray.vec);
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

glm::vec3 phong(const Hit &hit, const std::vector<Light> &lights,
    const std::vector<const Object *> &scene){
    const Material &mtl = hit.obj->mtl;

    glm::vec3 N = glm::normalize(hit.normal);
    glm::vec3 V = glm::normalize(hit.eye_dir);   // 表面 → 眼睛

    glm::vec3 color = mtl.Ka;

    for(const auto &l : lights){
        // 約定：l.dir 是「表面 → 光源」
        glm::vec3 L = glm::normalize(l.dir);

        Ray shadow_ray = Ray();
        shadow_ray.point = hit.pos;
        shadow_ray.vec = l.dir;
        Hit shadow_hit = firstHit(shadow_ray, scene);

        // Diffuse
        float ndotl = glm::max(0.0f, glm::dot(N, L));
        glm::vec3 diffuse = mtl.Kd * l.illum * ndotl;

        // Specular（注意入射向量要用 -L）
        glm::vec3 R = glm::reflect(L, N);
        float rv = glm::max(0.0f, glm::dot(R, V));
        float spec_pow = (mtl.exp > 0.0f) ? std::pow(rv, mtl.exp) : 0.0f;
        glm::vec3 specular = mtl.Ks * l.illum * spec_pow;

        if(!shadow_hit.hit){
            color += diffuse + specular;
        }
    }
    return color; // 寫圖前再做 tone mapping / gamma
}


glm::vec3 path_tracing(Ray ray,
    std::vector<const Object *> scene,
    std::vector<Light> &lights,
    int depth){
    if(depth == MAX_DPETH) return glm::vec3{ 0.0f };

    Hit h = firstHit(ray, scene);
    if(!h.hit) return glm::vec3(0.0f);
    if(h.kind == ObjKind::Sphere){
        // std::cout << h.pos << std::endl;
        // std::cout << ray.point + ray.vec << std::endl;
        // std::cout << "====" << std::endl;
    }

    glm::vec3 n = glm::normalize(h.normal);
    // glm::vec3 color = glm::vec3(255 * (h.pos.x + 1.0) / 2.0,
    //     255 * (h.pos.y + 1.0) / 2.0,
    //     255 * (h.pos.z + 1.0) / 2.0);
    // std::cout << h.pos << std::endl;
    glm::vec3 color = phong(h, lights, scene);


    float prob = get_rand();

    if(prob < h.obj->mtl.reflect || 1){
        glm::vec3 reflect_color{ 0.0f };
        // for(int i = 0; i < 1; i++){
        //     glm::vec3 reflect_vec = glm::reflect(ray.vec, n);
        //     glm::vec3 esp = glm::vec3(get_esp(), get_esp(), get_esp());
        //     reflect_vec += esp;
        //     reflect_vec = glm::normalize(reflect_vec);
        //     Ray reflect_ray = Ray(h.pos, reflect_vec);
        //     reflect_color += path_tracing(reflect_ray, scene, lights, depth + 1);
        // }
        // reflect_color = reflect_color / SAMPLE;
        // color = color + reflect_color;

        glm::vec3 reflect_vec = glm::reflect(ray.vec, n);
        Ray reflect_ray = Ray(h.pos, reflect_vec);
        reflect_color += path_tracing(reflect_ray, scene, lights, depth + 1);
        color = color + reflect_color * h.obj->mtl.reflect;

    }


    // std::cout << color << std::endl;
    return color;
}


GLuint tex = 0;
GLuint vao, vbo;
Shader *shader;
GLuint prog = 0;
void gen_texture(){
    // --- 放在 main() 開頭初始化完成後、進入渲染前 ---
// 解析度
    const int W = resolution.first;
    const int H = resolution.second;

    // 建立一張 RGB texture 當畫布
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // 先配好大小（不帶資料）
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    // 全螢幕 quad（NDC）
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    float quad_verts[] = {
        //   pos      //  uv
           -1.f, -1.f,  0.f, 0.f,
            1.f, -1.f,  1.f, 0.f,
            1.f,  1.f,  1.f, 1.f,
           -1.f,  1.f,  0.f, 1.f,
    };
    GLuint ebo;
    unsigned int idx[] = { 0,1,2, 0,2,3 };

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    // 你應該已有簡單的 shader；沒有的話做一個超簡單的：
    // VS: 把 pos 傳到 clip space，傳 uv
    // FS: 取樣 texture
    shader = new Shader("shader/shader.vs", "shader/shader.fs");
    prog = shader->ID; // 下面有提供
    glUseProgram(prog);
    GLint locPos = glGetAttribLocation(prog, "aPos");
    GLint locUV = glGetAttribLocation(prog, "aUV");
    glEnableVertexAttribArray(locPos);
    glVertexAttribPointer(locPos, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(locUV);
    glVertexAttribPointer(locUV, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (2 * sizeof(float)));
    glUseProgram(0);

}

int render_array[500 * 500] = {};


int main(int argc, char **argv){
    (void) argc; (void) argv;

    std::random_device rd;
    unsigned int seed = rd();
    mt_rand = std::mt19937(1234);

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

    gen_texture();

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

    std::shuffle(render_array, render_array + W * H, mt_rand);

    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // /*
        int end = std::min(W * H, pixel_cursor + k_pixels_per_frame);
        for(int p = pixel_cursor; p < end; ++p){
            int j = render_array[p] / W;              // row
            int i = render_array[p] % W;              // col

            glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f) + dy * (float(j) + 0.5f);
            glm::vec3 dir = glm::normalize(pixel_pos - cam_eye);

            Ray ray(cam_eye, dir);

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

            col = col / SAMPLE;

            col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f));
            // col = glm::pow(col, glm::vec3(1.0f / 2.2f)); // gamma

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
