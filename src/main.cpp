#include "GLinclude.h"
#include "glsl.h"
#include "bdpt_cu_helper.h"
#include "texture.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
// #include <corecrt_math_defines.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <ctime>
#include <omp.h>
#include <map>


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#define RENDER_THREADS 10
#define GROUPING 1
#define MAX_ITER 999999999
#define LIGHT_DEPTH 4
#define EYE_DEPTH 4

// Window size
glm::vec3 eye;
std::pair<int, int> resolution;
int render_array[600 * 600] = {};


void init_camera(Camera camera, float F,
    int W, int H,
    glm::vec3 &UL, glm::vec3 &dx, glm::vec3 &dy){
    float aspect = float(W) / float(H);
    float theta = F * PI / 180.0f;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;

    glm::vec3 w = glm::normalize(camera.eye - camera.look_at);
    glm::vec3 u = glm::normalize(glm::cross(camera.view_up, w));
    glm::vec3 v = glm::cross(w, u);

    UL = camera.eye - half_width * u + half_height * v - w;
    dx = (2 * half_width * u) / float(W);
    dy = (-2 * half_height * v) / float(H);
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
            input >> ka >> kd >> ks >> mtl.reflect >> mtl.refract >> mtl.exp;
            float total = ka + kd + ks;
            mtl.Kg = color * ka;
            mtl.Kd = color * kd;
            mtl.Ks = color * ks;
            mtl.glossy = ka;
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
    std::cout << "Trainagle:" << std::endl;
    std::cout << tri_cnt << std::endl;


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


    Camera ori_cam = camera;

    float F = 50;
    glm::vec3 UL, dx, dy;
    const glm::vec3 cam_eye = camera.eye;
    init_camera(camera, F, W, H, UL, dx, dy);
    CudaCamera cam = {};
    cam.eye = { cam_eye.x, cam_eye.y, cam_eye.z };
    cam.UL = { UL.x, UL.y, UL.z };
    cam.dx = { dx.x, dx.y, dx.z };
    cam.dy = { dy.x, dy.y, dy.z };

    std::vector<CudaLight> light(3);
    light[1].dir.x = 1.0f;
    light[1].dir.y = 0.0f;
    light[1].dir.z = 0.0f;
    light[1].pos.x = -0.49f;
    light[1].pos.y = .0f;
    light[1].pos.z = 0.0f;
    light[1].illum.x = 0.0f;
    light[1].illum.y = 1.0f;
    light[1].illum.z = 1.0f;
    light[1].cutoff = glm::radians(10.0f);
    light[2].dir.x = -1.0f;
    light[2].dir.y = -1.0f;
    light[2].dir.z = 0.0f;
    light[2].pos.x = 0.49f;
    light[2].pos.y = .49f;
    light[2].pos.z = 0.0f;
    light[2].illum.x = 1.0f;
    light[2].illum.y = 1.0f;
    light[2].illum.z = 0.0f;
    light[2].cutoff = glm::radians(50.0f);
    light[0].is_parallel = false;
    light[0].dir.x = 0.0f;
    light[0].dir.y = -1.0f;
    light[0].dir.z = 1.0f;
    light[0].illum.x = 20.0f;
    light[0].illum.y = 0.0f;
    light[0].illum.z = 20.0f;
    light[0].is_parallel = true;

    move_data_to_cuda(groups, light, 100);

    auto end_time = std::chrono::steady_clock::now();
    auto start_time = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);


    bool is_first = 1;
    bool stop_write = false;

    FILE *gp = _popen("gnuplot -persist", "w");
    fprintf(gp, "set term wxt\n");
    fprintf(gp, "set grid\n");
    fflush(gp);
    std::vector<float> rms_history;
    std::vector<CudaVec3> cuda_results(W * H);
    while(!glfwWindowShouldClose(window)){
        /*--------------------------
        Image Save Function
        --------------------------*/
        auto save_image = [&](int render_count){
            if(framebuffer.empty()){
                std::cerr << "[Error] Framebuffer is empty!" << std::endl;
                return;
            }
            cv::Mat img_rgb(H, W, CV_8UC3, (void *) framebuffer.data());

            cv::Mat img_bgr, img_bgr_Flip;

            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            cv::flip(img_bgr, img_bgr_Flip, 0);

            float current_rms = rms_history.empty() ? 0.0f : rms_history.back();

            std::stringstream ss;
            ss << "result_E" << EYE_DEPTH
                << "_L" << LIGHT_DEPTH
                << "_" << render_count
                << "_" << std::fixed << std::setprecision(4) << current_rms
                << ".png";

            std::string file_name = ss.str();
            std::cout << "[Save] " << file_name << std::endl;

            if(!cv::imwrite(file_name, img_bgr_Flip)){
                std::cerr << "[Error] Failed to save image: " << file_name << std::endl;
            }

            std::stringstream plot_file;
            plot_file << "plot_E" << EYE_DEPTH
                << "_L" << LIGHT_DEPTH
                << "_" << render_count
                << "_" << std::fixed << std::setprecision(4) << current_rms
                << ".png";
            std::string plot_filename = plot_file.str();
            fprintf(gp, "set terminal pngcairo size 800, 800 enhanced font 'Arial,12'\n");
            fprintf(gp, "set output '%s'\n", plot_filename.c_str());

            fprintf(gp, "plot '-' using 1:2 with lines title 'RMS'\n");
            for(int i = 0; i < (int) rms_history.size(); ++i){
                fprintf(gp, "%d %f\n", i, rms_history[i]);
            }
            fprintf(gp, "e\n");
            fflush(gp);
        };

        /*--------------------------
        GUI
        --------------------------*/
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glfwPollEvents();

        ImGui::Begin("F and F/");
        if(ImGui::Button("Save Image")){
            save_image(render_conut);
        }

        ImGui::End();

        glm::vec3 col(0.0f);


        /*--------------------------
        Ray Tracing Render Loop
        --------------------------*/
        if(render_conut < MAX_ITER){
            std::cout << "Render Iteration: " << render_conut + 1 << std::endl;
            render_conut++;

            std::cout << "Generate Eye Ray Elapsed time: " << diff.count() << " ms" << std::endl;
            start_time = std::chrono::steady_clock::now();
            end_time = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Generate Light Ray Elapsed time: " << diff.count() << " ms" << std::endl;
            cuda_results.resize(W * H);
            start_time = std::chrono::steady_clock::now();
            run_cuda_bdpt(cam, cuda_results.data(), EYE_DEPTH, LIGHT_DEPTH, W, H);
            float rms = 0;
            for(int i = 0; i < W; i++){
                for(int j = 0; j < H; j++){
                    CudaVec3 cu_col = cuda_results[j * W + i];
                    glm::vec3 col = glm::vec3(cu_col.x, cu_col.y, cu_col.z);

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
            end_time = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Render Elapsed time: " << diff.count() << " ms" << std::endl;
            std::cout << "Iteration：" << render_conut << " Error：" << rms << std::endl;
            is_first = 0;
        }
        else if(!stop_write){
            save_image(render_conut);
            stop_write = true;
        }
        // std::cout << "update" << std::endl;

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
    _pclose(gp);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
