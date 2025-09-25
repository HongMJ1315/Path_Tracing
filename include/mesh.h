#ifndef MESH_H
#define MESH_H

#include "GLinclude.h"
#include <string>
#include <vector>
#include <glm/glm.hpp>

class Mesh{
public:
    struct Material{
        std::string name;
        glm::vec3   Kd{ 1.0f,1.0f,1.0f };
        std::string map_Kd_path;
        GLuint      tex = 0;
    };
    struct DrawRange{
        GLsizei index_count = 0;
        GLsizei index_offset = 0;
        int     material_id = -1;
    };

    Mesh() = default;
    explicit Mesh(const std::string &obj_path){ load(obj_path); }
    ~Mesh();

    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&other) noexcept;
    Mesh &operator=(Mesh &&other) noexcept;

    bool load(const std::string &obj_path);

    bool load_from_memory(const std::vector<glm::vec3> &positions,
        const std::vector<glm::vec3> *normals,
        const std::vector<glm::vec2> *uvs,
        const std::vector<unsigned int> &indices,
        const glm::vec3 &solidKd = glm::vec3(1.0f),
        GLuint texture = 0);

    bool load_from_memory(const std::vector<glm::vec3> &positions,
        const std::vector<glm::vec3> *normals,
        const std::vector<glm::vec2> *uvs,
        const std::vector<glm::vec3> *colors,
        const std::vector<unsigned int> &indices,
        const glm::vec3 &solidKd = glm::vec3(1.0f),
        GLuint texture = 0);

    bool load_from_interleaved(const std::vector<float> &interleaved,
        const std::vector<unsigned int> &indices,
        const glm::vec3 &solidKd = glm::vec3(1.0f),
        GLuint texture = 0);

    void set_material(int i, const glm::vec3 &Kd, GLuint texture = 0);
    void draw() const;

    // 變換 / 狀態
    void set_model(const glm::mat4 &m){ model_ = m; }
    const glm::mat4 &model() const{ return model_; }
    const std::vector<Material> &materials() const{ return materials_; }
    bool ok() const{ return ok_; }

    // === 給 Ray TBO 用的 getters（新增） ===
    const std::vector<float> &vertices()     const{ return vertices_; }
    const std::vector<unsigned int> &indices()      const{ return indices_; }
    const std::vector<DrawRange> &ranges()       const{ return ranges_; }
    int                              stride_floats()const{ return stride_floats_; }

private:
    // GL
    GLuint vao_ = 0, vbo_ = 0, ebo_ = 0;

    // CPU data
    std::vector<float>         vertices_; // P(3) N(3) UV(2) [C(3)]
    std::vector<unsigned int>  indices_;
    std::vector<DrawRange>     ranges_;
    std::vector<Material>      materials_;
    glm::mat4                  model_{ 1.0f };
    bool ok_ = false;

    // layout
    int  stride_floats_ = 8;
    bool has_color_ = false;

    // internals
    static std::string read_text_file(const std::string &path);
    static std::string dir_of(const std::string &path);
    static std::string trim(const std::string &s);
    static std::vector<std::string> split_ws(const std::string &s);

    bool parse_obj(const std::string &obj_path, const std::string &obj_src);
    bool parse_mtl(const std::string &mtl_path, const std::string &mtl_src, const std::string &base_dir);
    GLuint load_texture(const std::string &path);
    void create_gl_buffers();
    void clear();
};

// 小球產生器（原樣保留）
static Mesh make_uv_sphere(int stacks, int slices, float r, const glm::vec3 &color){
    std::vector<glm::vec3> P, N; std::vector<glm::vec2> T; std::vector<unsigned> I;
    for(int i = 0; i <= stacks; ++i){
        float v = float(i) / stacks, phi = v * glm::pi<float>();
        float y = std::cos(phi), rr = std::sin(phi);
        for(int j = 0; j <= slices; ++j){
            float u = float(j) / slices, th = u * glm::two_pi<float>();
            glm::vec3 n(rr * std::cos(th), y, rr * std::sin(th));
            N.push_back(glm::normalize(n)); P.push_back(n * r); T.push_back({ u,v });
        }
    }
    auto idx = [&](int a, int b){ return a * (slices + 1) + b; };
    for(int i = 0; i < stacks; ++i) for(int j = 0; j < slices; ++j){
        int a = idx(i, j), b = idx(i + 1, j), c = idx(i, j + 1), d = idx(i + 1, j + 1);
        I.push_back(a); I.push_back(b); I.push_back(c);
        I.push_back(c); I.push_back(b); I.push_back(d);
    }
    Mesh m; m.load_from_memory(P, &N, &T, I, color, 0); return m;
}

#endif // MESH_H
    