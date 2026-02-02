#include "GLinclude.h"
#include "glsl.h"
#include "ppm_cu_helper.h"
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

#ifdef WINDOWS
#define popen _popen
#define pclose _pclose
#endif

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

    GLFWwindow *window = glfwCreateWindow(1800, 800, "Ray Tracing", nullptr, nullptr);
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
    std::vector<CudaLight> cu_light;
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
        else if(t == 'L'){
            CudaLight light;
            float cutoff_deg;
            std::cout << "read L" << std::endl;
            input >> light.pos >> light.dir >> light.illum >> cutoff_deg;
            light.cutoff = glm::radians(cutoff_deg);
            input >> light.is_parallel;
            cu_light.push_back(light);
        }
    }

    const int W = resolution.first;   // width
    const int H = resolution.second;  // height

    glViewport(0, 0, 1800, 600);
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
    std::cout << "Light:" << std::endl;
    std::cout << cu_light.size() << std::endl;

    for(auto l: cu_light){
        std::cout << "Light Dir: " << l.dir << " Pos: " << l.pos << " Illum: " << l.illum << " Cutoff: " << l.cutoff << " is_parallel: " << l.is_parallel << std::endl;
    }


    std::vector<const Object *> scene;
    scene.reserve(balls.size() + triangles.size());
    for(const auto &s : balls)      scene.push_back(&s);
    for(const auto &t : triangles)  scene.push_back(&t);

    std::vector<float>  ppm_buffer(H * W * 3, 0), bdpt_buffer(H * W * 3, 0);
    std::vector<unsigned char> framebuffer(W * H * 3 * 3, 0),
        last_ppm(W * H * 3, 0), last_bdpt(W * H * 3, 0),
        current_ppm(W * H * 3, 0), current_bdpt(W * H * 3, 0);
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

    move_data_to_cuda_ppm(groups, cu_light, 1000000);
    move_data_to_cuda_bdpt(groups, cu_light, 100);

    auto end_time = std::chrono::steady_clock::now();
    auto start_time = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);


    bool is_first = 1;
    bool stop_write = false;

    FILE *rms_gp = popen("gnuplot -persist", "w");
    if(rms_gp){
        fprintf(rms_gp, "set term wxt title 'RMS Convergence Comparison'\n");
        fprintf(rms_gp, "set grid\n");
        fprintf(rms_gp, "set xlabel 'Iteration'\n");
        fprintf(rms_gp, "set ylabel 'RMS Error'\n");
        fflush(rms_gp);
    }

    std::vector<float> ppm_rms_history, bdpt_rms_history;
    std::vector<CudaVec3> ppm_results(W * H);
    std::vector<CudaVec3> bdpt_results(W * H);
    while(!glfwWindowShouldClose(window)){
        /*--------------------------
        Image Save Function
        --------------------------*/
        auto save_image = [&](int render_count){
            auto img_process = [](std::vector<uchar> &results, int W, int H){
                cv::Mat img_rgb(H, W, CV_8UC3, (void *) results.data());
                cv::Mat img_bgr, img_bgr_Flip;
                cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);

                cv::flip(img_bgr, img_bgr_Flip, 0);

                return img_bgr_Flip;
            };
            if(framebuffer.empty()){
                std::cerr << "[Error] Framebuffer is empty!" << std::endl;
                return;
            }

            std::vector<uchar> combined(W * H * 3, 0);
            for(int i = 0; i < W * H; ++i){
                combined[i * 3 + 0] = std::min(((int) current_ppm[i * 3 + 0] + (int) current_bdpt[i * 3 + 0]) / 2, 255);
                combined[i * 3 + 1] = std::min(((int) current_ppm[i * 3 + 1] + (int) current_bdpt[i * 3 + 1]) / 2, 255);
                combined[i * 3 + 2] = std::min(((int) current_ppm[i * 3 + 2] + (int) current_bdpt[i * 3 + 2]) / 2, 255);
            }

            cv::Mat bdpt_img = img_process(current_bdpt, W, H);
            cv::Mat ppm_img = img_process(current_ppm, W, H);
            cv::Mat img_bgr_Flip = img_process(combined, W, H);
            std::stringstream ss;
            ss << "E" << EYE_DEPTH
                << "_L" << LIGHT_DEPTH
                << "_" << render_count
                << "_" << std::fixed << std::setprecision(4);
            std::stringstream ppm_ss;
            ppm_ss << "ppm_" << ss.str() << ppm_rms_history.back() << ".png";
            std::string ppm_file_name = ppm_ss.str();
            std::cout << "[Save] " << ppm_file_name << std::endl;

            if(!cv::imwrite(ppm_file_name, ppm_img)){
                std::cerr << "[Error] Failed to save image: " << ppm_file_name << std::endl;
            }

            std::stringstream bdpt_ss;
            bdpt_ss << "bdpt_" << ss.str() << bdpt_rms_history.back() << ".png";
            std::string bdpt_file_name = bdpt_ss.str();
            std::cout << "[Save] " << bdpt_file_name << std::endl;
            if(!cv::imwrite(bdpt_file_name, bdpt_img)){
                std::cerr << "[Error] Failed to save image: " << bdpt_file_name << std::endl;
            }
            std::stringstream combined_ss;
            combined_ss << "combined_" << ss.str() << ".png";
            std::string combined_file_name = combined_ss.str();
            std::cout << "[Save] " << combined_file_name << std::endl;
            if(!cv::imwrite(combined_file_name, img_bgr_Flip)){
                std::cerr << "[Error] Failed to save image: " << combined_file_name << std::endl;
            }

            std::stringstream plot_file;
            plot_file << "plot_Combined_E" << EYE_DEPTH << "_L" << LIGHT_DEPTH << "_" << render_count << ".png";
            std::string plot_filename = plot_file.str();

            fprintf(rms_gp, "set terminal pngcairo size 800, 600 enhanced font 'Arial,12'\n");
            fprintf(rms_gp, "set output '%s'\n", plot_filename.c_str());

            fprintf(rms_gp, "plot '-' using 1:2 with lines title 'PPM RMS', '-' using 1:2 with lines title 'BDPT RMS'\n");

            for(int i = 0; i < (int) ppm_rms_history.size(); ++i){
                fprintf(rms_gp, "%d %f\n", i, ppm_rms_history[i]);
            }
            fprintf(rms_gp, "e\n");

            for(int i = 0; i < (int) bdpt_rms_history.size(); ++i){
                fprintf(rms_gp, "%d %f\n", i, bdpt_rms_history[i]);
            }
            fprintf(rms_gp, "e\n");

            fflush(rms_gp);

            fprintf(rms_gp, "unset output\n");
            fprintf(rms_gp, "set terminal wxt\n");
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

            ppm_results.resize(W * H);
            bdpt_results.resize(W * H);
            start_time = std::chrono::steady_clock::now();
            run_cuda_ppm(cam, ppm_results.data(), EYE_DEPTH, LIGHT_DEPTH, W, H);
            end_time = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "PPM Render Elapsed time: " << diff.count() << " ms" << std::endl;
            start_time = std::chrono::steady_clock::now();
            run_cuda_bdpt(cam, bdpt_results.data(), EYE_DEPTH, LIGHT_DEPTH, W, H);
            end_time = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "BDPT Render Elapsed time: " << diff.count() << " ms" << std::endl;
            float ppm_rms = 0, bdpt_rms = 0;
            int total_width = W * 3;

            for(int i = 0; i < W; i++){
                for(int j = 0; j < H; j++){
                    CudaVec3 ppm_cu_col = ppm_results[j * W + i];
                    glm::vec3 ppm_col = glm::vec3(ppm_cu_col.x, ppm_cu_col.y, ppm_cu_col.z);
                    ppm_col = glm::clamp(ppm_col, glm::vec3(0.0f), glm::vec3(1.0f));
                    ppm_col = glm::pow(ppm_col, glm::vec3(1.0f / 2.2f));
                    float ppm[3] = { ppm_col.x, ppm_col.y, ppm_col.z };

                    CudaVec3 bdpt_cu_col = bdpt_results[j * W + i];
                    glm::vec3 bdpt_col = glm::vec3(bdpt_cu_col.x, bdpt_cu_col.y, bdpt_cu_col.z);
                    bdpt_col = glm::clamp(bdpt_col, glm::vec3(0.0f), glm::vec3(1.0f));
                    bdpt_col = glm::pow(bdpt_col, glm::vec3(1.0f / 2.2f));
                    float bdpt[3] = { bdpt_col.x, bdpt_col.y, bdpt_col.z };

                    int row_flipped = H - 1 - j; // Y軸翻轉
                    int image_idx = (size_t(row_flipped) * W + i) * 3;

                    int ppm_frame_idx = (size_t(row_flipped) * total_width + i) * 3;
                    int bdpt_frame_idx = (size_t(row_flipped) * total_width + (i + W)) * 3;
                    int combined_frame_idx = (size_t(row_flipped) * total_width + (i + W * 2)) * 3;

                    ppm_buffer[image_idx + 0] += ppm_col.r;
                    ppm_buffer[image_idx + 1] += ppm_col.g;
                    ppm_buffer[image_idx + 2] += ppm_col.b;

                    current_ppm[image_idx + 0] = (unsigned char) ((ppm_buffer[image_idx + 0] / (float) render_conut) * 255.f);
                    current_ppm[image_idx + 1] = (unsigned char) ((ppm_buffer[image_idx + 1] / (float) render_conut) * 255.f);
                    current_ppm[image_idx + 2] = (unsigned char) ((ppm_buffer[image_idx + 2] / (float) render_conut) * 255.f);

                    framebuffer[ppm_frame_idx + 0] = current_ppm[image_idx + 0];
                    framebuffer[ppm_frame_idx + 1] = current_ppm[image_idx + 1];
                    framebuffer[ppm_frame_idx + 2] = current_ppm[image_idx + 2];

                    bdpt_buffer[image_idx + 0] += bdpt_col.r;
                    bdpt_buffer[image_idx + 1] += bdpt_col.g;
                    bdpt_buffer[image_idx + 2] += bdpt_col.b;

                    current_bdpt[image_idx + 0] = (unsigned char) ((bdpt_buffer[image_idx + 0] / (float) render_conut) * 255.f);
                    current_bdpt[image_idx + 1] = (unsigned char) ((bdpt_buffer[image_idx + 1] / (float) render_conut) * 255.f);
                    current_bdpt[image_idx + 2] = (unsigned char) ((bdpt_buffer[image_idx + 2] / (float) render_conut) * 255.f);

                    framebuffer[bdpt_frame_idx + 0] = current_bdpt[image_idx + 0];
                    framebuffer[bdpt_frame_idx + 1] = current_bdpt[image_idx + 1];
                    framebuffer[bdpt_frame_idx + 2] = current_bdpt[image_idx + 2];

                    framebuffer[combined_frame_idx + 0] = (unsigned char) ((int) (current_ppm[image_idx + 0] + (int) current_bdpt[image_idx + 0]) / 2.f);
                    framebuffer[combined_frame_idx + 1] = (unsigned char) ((int) (current_ppm[image_idx + 1] + (int) current_bdpt[image_idx + 1]) / 2.f);
                    framebuffer[combined_frame_idx + 2] = (unsigned char) ((int) (current_ppm[image_idx + 2] + (int) current_bdpt[image_idx + 2]) / 2.f);

                    if(!is_first){
                        for(int k = 0; k < 3; k++){

                            ppm_rms += std::powf(current_ppm[image_idx + k] - last_ppm[image_idx + k], 2);
                            bdpt_rms += std::powf(current_bdpt[image_idx + k] - last_bdpt[image_idx + k], 2);
                        }
                    }
                    last_ppm[image_idx + 0] = current_ppm[image_idx + 0];
                    last_ppm[image_idx + 1] = current_ppm[image_idx + 1];
                    last_ppm[image_idx + 2] = current_ppm[image_idx + 2];
                    last_bdpt[image_idx + 0] = current_bdpt[image_idx + 0];
                    last_bdpt[image_idx + 1] = current_bdpt[image_idx + 1];
                    last_bdpt[image_idx + 2] = current_bdpt[image_idx + 2];
                }
            }
            ppm_rms = std::sqrt(ppm_rms) / 255.0f;
            ppm_rms_history.push_back(ppm_rms);

            bdpt_rms = std::sqrt(bdpt_rms) / 255.0f;
            bdpt_rms_history.push_back(bdpt_rms);
            fprintf(rms_gp, "plot '-' using 1:2 with lines lw 1.5 title 'PPM RMS', '-' using 1:2 with lines lw 1.5 title 'BDPT RMS'\n");

            // 第一組：PPM
            for(int i = 0; i < (int) ppm_rms_history.size(); ++i){
                fprintf(rms_gp, "%d %f\n", i, ppm_rms_history[i]);
            }
            fprintf(rms_gp, "e\n");

            // 第二組：BDPT
            for(int i = 0; i < (int) bdpt_rms_history.size(); ++i){
                fprintf(rms_gp, "%d %f\n", i, bdpt_rms_history[i]);
            }
            fprintf(rms_gp, "e\n");

            fflush(rms_gp);

            std::cout << "Iteration：" << render_conut << std::endl;
            std::cout << "PPM Error：" << ppm_rms << std::endl;
            std::cout << "BDPT Error：" << bdpt_rms << std::endl;
            std::cout << "------------------------" << std::endl;
            is_first = 0;
        }
        else if(!stop_write){
            save_image(render_conut);
            stop_write = true;
        }
        // std::cout << "update" << std::endl;

        // */
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W * 3, H, GL_RGB, GL_UNSIGNED_BYTE, framebuffer.data());


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
    pclose(rms_gp);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
