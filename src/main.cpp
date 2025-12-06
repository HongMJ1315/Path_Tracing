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
#include <opencv2/core/utils/logger.hpp>
#include <ctime>
#include <omp.h>
#include <map>


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#define RENDER_THREADS 5
#define GROUPING 1

// Window size
glm::vec3 eye;
std::pair<int, int> resolution;
int render_array[600 * 600] = {};


//run type = 0 Use FOV, type = 1 Use f_mm 
void get_camera(int W, int H, float f_mm, Camera &camera,
    glm::vec3 &UL, glm::vec3 &dx, glm::vec3 &dy, bool run_type,
    float sensor_w = 36.0f, float sensor_h = 24.0f){
    auto deg2rad = [](float d){ return d * float(M_PI) / 180.0f; };
    auto rad2deg = [](float r){ return r * 180.0f / float(M_PI); };
    float vfov = deg2rad(camera.fov);
    if(run_type){

        vfov = 2.0f * std::atan(sensor_h / (2.0f * f_mm));
        camera.fov = rad2deg(vfov);
    }
    const float fov_rad = vfov;
    const float aspect = float(W) / float(H);

    const glm::vec3 cam_eye = camera.eye;
    glm::vec3 f = glm::normalize(camera.look_at - camera.eye);
    glm::vec3 up = glm::normalize(camera.view_up);
    glm::vec3 r = glm::normalize(glm::cross(f, up));
    glm::vec3 u = glm::cross(r, f);

    const float dist = 1.0f;
    const glm::vec3 C = cam_eye + dist * f;
    const float half_h = std::tan(0.5f * fov_rad);
    const float half_w = half_h * aspect;

    UL = C + u * half_h - r * half_w;
    glm::vec3 UR = C + u * half_h + r * half_w;
    glm::vec3 LL = C - u * half_h - r * half_w;

    dx = (UR - UL) / float(W);
    dy = (LL - UL) / float(H);
}

inline glm::vec2 concentric_sample_disk(float u1, float u2){
    // [0,1)^2 -> [-1,1]^2
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    if(sx == 0 && sy == 0) return { 0,0 };

    float r, theta;
    if(std::abs(sx) > std::abs(sy)){
        r = sx;
        theta = (float) M_PI / 4.0f * (sy / sx);
    }
    else{
        r = sy;
        theta = (float) M_PI / 2.0f - (float) M_PI / 4.0f * (sx / sy);
    }
    return { r * std::cos(theta), r * std::sin(theta) };
}

void gen_eyeray(std::vector<EyeRayInfo> &eyeray, Camera ori_cam,
    const int W, const int H, glm::vec3 UL, glm::vec3 dx, glm::vec3 dy){
    eyeray.clear();
    for(int p = 0; p < W * H; p++){
        int j = render_array[p] / W;              // row
        int i = render_array[p] % W;              // col
        for(int s = 0; s < SAMPLE; s++){
            // std::cout << ori_cam.fov << std::endl;
            float jx = rng_uniform01() - 0.5f;
            float jy = rng_uniform01() - 0.5f;
            glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f + jx) + dy * (float(j) + 0.5f + jy);

            glm::vec3 ray_dir = glm::normalize(pixel_pos - ori_cam.eye);
            Ray ray(ori_cam.eye, ray_dir, 1, RayType::EYE);
            EyeRayInfo info;
            info.i = i, info.j = j;
            info.ray = ray;

            eyeray.push_back(info);
        }
    }
}


std::map<int, AABB> groups;
int main(int argc, char **argv){
    (void) argc; (void) argv;
    omp_set_num_threads(RENDER_THREADS);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

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

    GLFWwindow *window = glfwCreateWindow(800, 800, "Ray Tracing", nullptr, nullptr);
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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 400");

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
    int group_id = 0;
    int obj_id = 0;
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
            Sphere *s = new Sphere;
            input >> s->center >> s->r;
            s->scale = glm::vec3(1, 1, 1);
            s->mtl = mtl;
            s->obj_id = obj_id++;
            // groups[group_id].add_obj(s);
            if(GROUPING)
                groups[group_id].add_obj(s);
            else
                groups[group_id++].add_obj(s);
            // balls.push_back(s);
            // std::cout << "read S" << std::endl;
        }
        else if(t == 'T'){
            Triangle *tri = new Triangle;
            glm::vec3 vert;
            for(int i = 0; i < 3; i++){
                input >> vert;
                tri->vert[i] = vert;
            }
            tri->mtl = mtl;
            tri->obj_id = obj_id++;
            if(GROUPING)
                groups[group_id].add_obj(tri);
            else
                groups[group_id++].add_obj(tri);
            // triangles.push_back(tri);
            // std::cout << "read T" << std::endl;
        }
        else if(t == 'M'){
            float ka, kd, ks;
            glm::vec3 color;
            input >> color;
            input >> ka >> kd >> ks >> mtl.reflect >> mtl.refract;
            mtl.Kg = color * ka;
            mtl.Kd = color * kd;
            mtl.Ks = color * ks;
            // std::cout << color << " " << mtl.kg << " " << mtl.Kd << " " << mtl.Ks << std::endl;
            // std::cout << "read M" << std::endl;
        }
        else if(t == 'L'){
            Light light;
            input >> light.dir;
            lights.push_back(light);
        }
        else if(t == 'G' && GROUPING){
            input >> group_id;
        }
        else if(t == '/'){
            input >> t;
            if(t == '/'){
                std::string trash;
                std::getline(input, trash);
                continue;
            }
        }
    }

    const int W = resolution.first;   // width
    const int H = resolution.second;  // height

    glViewport(0, 0, 600, 600);
    glDisable(GL_DEPTH_TEST);

    gen_texture(resolution.first, resolution.second);

    GLuint vao = get_vao(), prog = get_shader(), tex = get_texture();

    int ball_cnt = 0, tri_cnt = 0;
    for(auto g : groups){
        for(auto i : g.second.objs){
            const Sphere *sph = dynamic_cast<const Sphere *>(i);
            const Triangle *tri = dynamic_cast<const Triangle *>(i);
            if(sph) ball_cnt++;
            else tri_cnt++;
        }
    }

    std::cout << "VAO: " << vao << std::endl;
    std::cout << "program: " << prog << std::endl;
    std::cout << "Eye Position: " << camera.eye << std::endl;
    std::cout << "Screen Info: " << std::endl;
    std::cout << "  Look At: " << camera.look_at << std::endl;
    std::cout << "  View Up: " << camera.view_up << std::endl;
    std::cout << "  FOV: " << camera.fov << std::endl;
    std::cout << "Ball:" << std::endl;
    std::cout << ball_cnt << std::endl;

    // for(auto b : balls){
    //     std::cout << "  Location: " << b.center << std::endl;
    //     std::cout << "  R: " << b.r << std::endl;
    // }
    std::cout << "Trainagle:" << std::endl;
    std::cout << tri_cnt << std::endl;

    // std::cout << "Bounding location: " << std::endl;
    // for(auto i : groups){
    //     std::cout << i.second.min << std::endl;
    //     std::cout << i.second.max << std::endl;
    //     std::cout << i.second.objs.size() << std::endl;
    //     std::cout << std::endl;
    // }
    // for(auto t : triangles){
    //     std::cout << "  Vertex: (" << std::endl;
    //     for(int i = 0; i < 3; i++){
    //         std::cout << "    vertex " << i << ": " << t.vert[i] << std::endl;
    //     }
    //     std::cout << "  )" << std::endl;
    // }




    std::vector<const Object *> scene;
    scene.reserve(balls.size() + triangles.size());
    for(const auto &s : balls)      scene.push_back(&s);
    for(const auto &t : triangles)  scene.push_back(&t);

    std::vector<float>  acc_buffer(H * W * 3, 0);
    std::vector<unsigned char> framebuffer(W * H * 3, 0), last_frame(W * H * 3, 0);
    int render_conut = 0;

    int pixel_cursor = 0;
    const int k_pixels_per_frame = 5;

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    glUseProgram(0);

    bool is_writed = false;

    bool is_depth = false;
    bool last_is_depth = is_depth;

    for(int i = 0; i < W * H; i++){
        render_array[i] = i;
    }

    Camera ori_cam = camera;

    float F = 50, A = 32;
    float last_F = F, last_A = A;
    glm::vec3 UL, dx, dy;
    const glm::vec3 cam_eye = camera.eye;
    get_camera(W, H, F, camera, UL, dx, dy, is_depth);

    glm::vec3 camF = glm::normalize(camera.look_at - camera.eye);
    glm::vec3 camR = glm::normalize(glm::cross(camF, glm::normalize(camera.view_up)));
    glm::vec3 camU = glm::cross(camR, camF);

    float focus_dist = glm::length(camera.look_at - camera.eye);
    float last_fd = focus_dist;

    float lens_mm = F / A * 0.5f;    // mm;

    std::random_device rd;
    unsigned int seed = rd();
    std::mt19937 mt_rand = std::mt19937(seed);
    std::shuffle(render_array, render_array + W * H, mt_rand);


    std::vector<EyeRayInfo> eyeray;
    gen_eyeray(eyeray, ori_cam, W, H, UL, dx, dy);

    init_eyeray(groups, eyeray, W, H);
    init_lightray(groups);
    // init_light_group();

    int progress = 0;
    int total_pixel = resolution.first * resolution.second;
    int percent = total_pixel / 100;



    bool is_first = 1;


    FILE *gp = _popen("gnuplot -persist", "w");
    fprintf(gp, "set term wxt\n");
    fprintf(gp, "set grid\n");
    fflush(gp);
    std::vector<float> rms_history;

    while(!glfwWindowShouldClose(window)){
        auto start_time = std::chrono::steady_clock::now();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glfwPollEvents();

        ImGui::Begin("F and F/");
        ImGui::SliderFloat("F", &F, 14.f, 200.0f);
        ImGui::SliderFloat("F/", &A, 1.4f, 32.f);
        ImGui::SliderFloat("Focus Dist", &focus_dist, 0.1f, 10.0f); // 可視需求調整範圍
        ImGui::Checkbox("Depth Field", &is_depth);
        if(ImGui::Button("Save Image")){
            cv::Mat img_rgb(H, W, CV_8UC3, framebuffer.data());
            cv::Mat img_bgr, img_bgr_Flip;

            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            cv::flip(img_bgr, img_bgr_Flip, 0);

            if(!cv::imwrite("color_output.png", img_bgr_Flip)){
                std::cerr << "彩色圖片輸出失敗！" << std::endl;
            }
        }

        ImGui::End();
        // /*
        if(is_depth && (last_A != A || last_F != F || last_fd != focus_dist)){
            pixel_cursor = 0;
            is_writed = 0;
            last_A = A; last_F = F; last_fd = focus_dist;
            std::fill(framebuffer.begin(), framebuffer.end(), 0);

            get_camera(W, H, F, camera, UL, dx, dy, 1);

            start_time = std::chrono::steady_clock::now();
        }
        else if(is_depth != last_is_depth){
            pixel_cursor = 0;
            is_writed = 0;
            std::fill(framebuffer.begin(), framebuffer.end(), 0);

            last_is_depth = is_depth;
            get_camera(W, H, F, ori_cam, UL, dx, dy, 0);
            start_time = std::chrono::steady_clock::now();
        }


        int end = std::min(W * H, pixel_cursor + k_pixels_per_frame);

        glm::vec3 col(0.0f);
        if(is_depth){
#pragma omp parallel for schedule(dynamic, 128)
            for(int p = pixel_cursor; p < end; ++p){
                int j = render_array[p] / W;              // row
                int i = render_array[p] % W;              // col


                float lens_radius = (F / A) * 0.5f * 0.1f;
                // std::cout << lens_radius << std::endl;
                for(int k = 0; k < SAMPLE; k++){
                    float jx = rng_uniform01() - 0.5f;
                    float jy = rng_uniform01() - 0.5f;

                    glm::vec3 pixel_pos = UL + dx * (float(i) + 0.5f + jx) + dy * (float(j) + 0.5f + jy);

                    glm::vec3 pin_dir = glm::normalize(pixel_pos - camera.eye);

                    float t_focus = focus_dist / glm::dot(pin_dir, camF);
                    glm::vec3 focus_point = camera.eye + pin_dir * t_focus;

                    glm::vec2 disk = concentric_sample_disk(rng_uniform01(), rng_uniform01());
                    glm::vec3 lens_offset = (disk.x * camR + disk.y * camU) * lens_radius;

                    glm::vec3 ray_origin = camera.eye + lens_offset;
                    glm::vec3 ray_dir = glm::normalize(focus_point - ray_origin);

                    Ray ray(ray_origin, ray_dir, 0, RayType::EYE);

                    col += path_tracing(ray, groups, lights, 0);
                    col = col / SAMPLE;
                }
                col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f));
                col = glm::pow(col, glm::vec3(1.0f / 2.2f)); // gamma

                size_t idx = (size_t(H - 1 - j) * W + i) * 3;
                framebuffer[idx + 0] = (unsigned char) (col.r * 255.0f);
                framebuffer[idx + 1] = (unsigned char) (col.g * 255.0f);
                framebuffer[idx + 2] = (unsigned char) (col.b * 255.0f);
            }

            // std::cout << col << std::endl;
            // if(p % percent == 0){
            //     std::cout << "Render " << progress++ << "%" << std::endl;
            // }
        }
        else{
            // col = eye_light_connect(i, j, groups);
            // col = light_debuger(i, j, UL, dx, dy, groups, ori_cam);
            gen_eyeray(eyeray, ori_cam, W, H, UL, dx, dy);
            render_conut++;
            init_eyeray(groups, eyeray, W, H);
            init_lightray(groups);
            std::vector<glm::vec3> cuda_results;
            cuda_results = run_cuda_eye_light_connect(W, H, groups);
            float rms = 0;
            for(int i = 0; i < W; i++){
                for(int j = 0; j < H; j++){
                    glm::vec3 col = cuda_results[j * W + i];

                    col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f));
                    col = glm::pow(col, glm::vec3(1.0f / 2.2f)); // gamma

                    int idx = (size_t(H - 1 - j) * W + i) * 3;
                    acc_buffer[idx + 0] += col.r;
                    acc_buffer[idx + 1] += col.g;
                    acc_buffer[idx + 2] += col.b;

                    framebuffer[idx + 0] = (unsigned char) ((acc_buffer[idx + 0] / (float) render_conut) * 255.f);
                    framebuffer[idx + 1] = (unsigned char) ((acc_buffer[idx + 1] / (float) render_conut) * 255.f);
                    framebuffer[idx + 2] = (unsigned char) ((acc_buffer[idx + 2] / (float) render_conut) * 255.f);
                    if(!is_first){
                        for(int k = 0; k < 3; k++){
                            rms += (framebuffer[idx + k] - last_frame[idx + k]) * (framebuffer[idx + k] - last_frame[idx + k]);
                        }
                    }
                    last_frame[idx + 0] = framebuffer[idx + 0];
                    last_frame[idx + 1] = framebuffer[idx + 1];
                    last_frame[idx + 2] = framebuffer[idx + 2];

                }
            }
            rms = std::sqrt(rms) / 255.f;
            rms_history.push_back(rms);
            // std::cout << rms << std::endl;
            fprintf(gp, "plot '-' using 1:2 with lines title 'RMS'\n");
            for(int i = 0; i < (int) rms_history.size(); ++i){
                fprintf(gp, "%d %f\n", i, rms_history[i]);
            }
            fprintf(gp, "e\n");
            fflush(gp);        // 讓 gnuplot 立即更新
            auto end_time = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Elapsed time: " << diff.count() << " ms" << std::endl;
            std::cout << "Iteration：" << render_conut << " Error：" << rms << std::endl;
            is_first = 0;
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

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
