#include "mesh.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <map>
#include <unordered_map>
#include <tuple>
#include <filesystem>

// stb_image for loading textures
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include <stb_image.h>

using std::string;
namespace fs = std::filesystem;

// pack v/vt/vn (with negative allowed) into a key
static inline uint64_t pack_key(int v, int vt, int vn){
    auto to20 = [](int x){ return (uint64_t) ((int64_t) x + (1ll << 19)) & 0xFFFFFll; };
    return (to20(v) << 40) | (to20(vt) << 20) | (to20(vn) << 0);
}

Mesh::~Mesh(){ clear(); }

Mesh::Mesh(Mesh &&other) noexcept{
    vao_ = other.vao_; other.vao_ = 0;
    vbo_ = other.vbo_; other.vbo_ = 0;
    ebo_ = other.ebo_; other.ebo_ = 0;
    vertices_ = std::move(other.vertices_);
    indices_ = std::move(other.indices_);
    ranges_ = std::move(other.ranges_);
    materials_ = std::move(other.materials_);
    model_ = other.model_;
    ok_ = other.ok_;
    stride_floats_ = other.stride_floats_;
    has_color_ = other.has_color_;
}
Mesh &Mesh::operator=(Mesh &&other) noexcept{
    if(this != &other){
        clear();
        vao_ = other.vao_; other.vao_ = 0;
        vbo_ = other.vbo_; other.vbo_ = 0;
        ebo_ = other.ebo_; other.ebo_ = 0;
        vertices_ = std::move(other.vertices_);
        indices_ = std::move(other.indices_);
        ranges_ = std::move(other.ranges_);
        materials_ = std::move(other.materials_);
        model_ = other.model_;
        ok_ = other.ok_;
        stride_floats_ = other.stride_floats_;
        has_color_ = other.has_color_;
    }
    return *this;
}

void Mesh::clear(){
    if(ebo_){ glDeleteBuffers(1, &ebo_); ebo_ = 0; }
    if(vbo_){ glDeleteBuffers(1, &vbo_); vbo_ = 0; }
    if(vao_){ glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    for(auto &m : materials_){
        if(m.tex){ glDeleteTextures(1, &m.tex); m.tex = 0; }
    }
    vertices_.clear();
    indices_.clear();
    ranges_.clear();
    materials_.clear();
    ok_ = false;
}

bool Mesh::load(const std::string &obj_path){
    clear();
    string src = read_text_file(obj_path);
    if(src.empty()){
        std::cerr << "[Mesh] Failed to read OBJ: " << obj_path << "\n";
        return false;
    }
    if(!parse_obj(obj_path, src)){
        std::cerr << "[Mesh] Failed to parse OBJ: " << obj_path << "\n";
        return false;
    }
    // OBJ 預設 stride = 8（P3 N3 UV2），沒有頂點色
    stride_floats_ = 8;
    has_color_ = false;
    create_gl_buffers();
    ok_ = true;
    return true;
}

void Mesh::create_gl_buffers(){
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), vertices_.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), indices_.data(), GL_STATIC_DRAW);

    GLsizei stride = stride_floats_ * sizeof(float);
    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *) 0);
    glEnableVertexAttribArray(1); // normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(2); // uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void *) (6 * sizeof(float)));
    if(has_color_){
        glEnableVertexAttribArray(3); // color（可選）
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void *) (8 * sizeof(float)));
    }
    glBindVertexArray(0);
}

void Mesh::draw() const{
    if(!ok_ || vao_ == 0) return;
    glBindVertexArray(vao_);

    for(const auto &r : ranges_){
        bool hasTex = false;
        glm::vec3 kd(1.0f);
        if(r.material_id >= 0 && r.material_id < (int) materials_.size()){
            const auto &m = materials_[r.material_id];
            kd = m.Kd;
            if(m.tex){
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, m.tex);
                hasTex = true;
            }
            else{
                glBindTexture(GL_TEXTURE_2D, 0);
            }
        }
        else{
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // 設定當前 program 的 uHasTex / uKd
        GLint prog = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
        if(prog != 0){
            GLint locHas = glGetUniformLocation((GLuint) prog, "uHasTex");
            if(locHas >= 0) glUniform1i(locHas, hasTex ? 1 : 0);
            GLint locKd = glGetUniformLocation((GLuint) prog, "uKd");
            if(locKd >= 0) glUniform3fv(locKd, 1, &kd.x);
        }

        glDrawElements(GL_TRIANGLES, r.index_count, GL_UNSIGNED_INT,
            (void *) (sizeof(unsigned int) * r.index_offset));
    }
    glBindVertexArray(0);
}

// ------------------------- OBJ / MTL parsing -------------------------

static bool parse_face_vertex(const std::string &token, int &vi, int &ti, int &ni){
    // 支援 v, v/vt, v//vn, v/vt/vn
    vi = ti = ni = 0;
    if(token.find('/') == string::npos){
        vi = std::stoi(token);
        return true;
    }
    std::string::size_type firstSlash = token.find('/');
    std::string::size_type secondSlash = token.find('/', firstSlash + 1);
    try{
        string s0 = token.substr(0, firstSlash);
        string s1 = (secondSlash == string::npos) ? token.substr(firstSlash + 1)
            : token.substr(firstSlash + 1, secondSlash - firstSlash - 1);
        string s2 = (secondSlash == string::npos) ? "" : token.substr(secondSlash + 1);
        vi = s0.empty() ? 0 : std::stoi(s0);
        ti = s1.empty() ? 0 : std::stoi(s1);
        ni = s2.empty() ? 0 : std::stoi(s2);
        return true;
    }
    catch(...){
        return false;
    }
}

bool Mesh::parse_obj(const std::string &obj_path, const std::string &obj_src){
    std::stringstream in(obj_src);
    string line;

    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> nor;
    std::vector<glm::vec2> uv;

    std::map<string, int> mtl_index;
    int current_mat = -1;

    std::unordered_map<uint64_t, unsigned int> vmap;

    struct Tri{ unsigned int i0, i1, i2; int mat; };
    std::vector<Tri> tris;

    string base = dir_of(obj_path);

    while(std::getline(in, line)){
        line = trim(line);
        if(line.empty() || line[0] == '#') continue;
        auto tokens = split_ws(line);
        if(tokens.empty()) continue;

        const string &t0 = tokens[0];
        if(t0 == "v" && tokens.size() >= 4){
            pos.emplace_back(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if(t0 == "vn" && tokens.size() >= 4){
            nor.emplace_back(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if(t0 == "vt" && tokens.size() >= 3){
            uv.emplace_back(std::stof(tokens[1]), std::stof(tokens[2]));
        }
        else if(t0 == "f" && tokens.size() >= 4){
            auto make_index = [&](const string &tok)->unsigned int{
                int vi, ti, ni;
                if(!parse_face_vertex(tok, vi, ti, ni)) return 0;
                auto fix_index = [](int idx, int n)->int{
                    if(idx > 0) return idx - 1;
                    if(idx < 0) return n + idx;
                    return -1;
                };
                int vix = fix_index(vi, (int) pos.size());
                int tix = fix_index(ti, (int) uv.size());
                int nix = fix_index(ni, (int) nor.size());
                uint64_t key = pack_key(vix, tix, nix);
                auto it = vmap.find(key);
                if(it != vmap.end()) return it->second;
                glm::vec3 P = (vix >= 0) ? pos[vix] : glm::vec3(0);
                glm::vec3 N = (nix >= 0) ? nor[nix] : glm::vec3(0, 0, 1);
                glm::vec2 T = (tix >= 0) ? uv[tix] : glm::vec2(0);
                unsigned int newIndex = (unsigned int) (vertices_.size() / 8);
                vertices_.push_back(P.x); vertices_.push_back(P.y); vertices_.push_back(P.z);
                vertices_.push_back(N.x); vertices_.push_back(N.y); vertices_.push_back(N.z);
                vertices_.push_back(T.x); vertices_.push_back(T.y);
                vmap[key] = newIndex;
                return newIndex;
            };
            unsigned int i0 = make_index(tokens[1]);
            for(size_t i = 2; i + 1 < tokens.size(); ++i){
                unsigned int i1 = make_index(tokens[i]);
                unsigned int i2 = make_index(tokens[i + 1]);
                tris.push_back({ i0,i1,i2,current_mat });
            }
        }
        else if(t0 == "usemtl" && tokens.size() >= 2){
            string m = tokens[1];

            if(mtl_index.count(m) == 0){
                // 先在已存在的 materials_ 中找同名（可能是 mtllib 先載入的）
                int found = -1;
                for(int i = 0; i < (int) materials_.size(); ++i){
                    if(materials_[i].name == m){ found = i; break; }
                }
                if(found >= 0){
                    // 已有同名材質（可能含貼圖），直接使用它
                    mtl_index[m] = found;
                }
                else{
                    // 沒找到才建立 placeholder，之後 parse_mtl() 會補齊
                    Material mat; mat.name = m;
                    mtl_index[m] = (int) materials_.size();
                    materials_.push_back(mat);
                }
            }
            current_mat = mtl_index[m];
        }
        else if(t0 == "mtllib" && tokens.size() >= 2){
            for(size_t i = 1; i < tokens.size(); ++i){
                string mtlfile = tokens[i];
                string mpath = (fs::path(base) / fs::path(mtlfile)).string();
                string msrc = read_text_file(mpath);
                if(msrc.empty()){
                    std::cerr << "[Mesh] Warning: cannot read MTL: " << mpath << "\n";
                }
                else{
                    parse_mtl(mpath, msrc, base);
                }
            }
        }
    }

    std::map<int, std::vector<unsigned int>> byMat;
    for(const auto &t : tris){
        byMat[t.mat].push_back(t.i0);
        byMat[t.mat].push_back(t.i1);
        byMat[t.mat].push_back(t.i2);
    }

    GLsizei offset = 0;
    for(auto &kv : byMat){
        const auto &idxs = kv.second;
        DrawRange r;
        r.material_id = kv.first;
        r.index_offset = offset;
        r.index_count = static_cast<GLsizei>(idxs.size());
        ranges_.push_back(r);
        indices_.insert(indices_.end(), idxs.begin(), idxs.end());
        offset += static_cast<GLsizei>(r.index_count);
    }
    return true;
}

// ★ 修好：newmtl 會更新既有 placeholder（與 usemtl 對齊），不再重複新增
bool Mesh::parse_mtl(const std::string & /*mtl_path*/,
    const std::string &mtl_src,
    const std::string &base_dir){
    std::istringstream in(mtl_src);
    string line;
    int current = -1;

    // 建 name->index 映射（包含 usemtl 時建立的 placeholder）
    std::unordered_map<std::string, int> name2idx;
    for(int i = 0; i < (int) materials_.size(); ++i){
        if(!materials_[i].name.empty()) name2idx[materials_[i].name] = i;
    }

    while(std::getline(in, line)){
        line = trim(line);
        if(line.empty() || line[0] == '#') continue;
        auto tok = split_ws(line);
        if(tok.empty()) continue;
        const string &t0 = tok[0];

        if(t0 == "newmtl" && tok.size() >= 2){
            const std::string &nm = tok[1];
            auto it = name2idx.find(nm);
            if(it != name2idx.end()){
                current = it->second;        // 更新既有
            }
            else{
                Material m; m.name = nm;     // 新增
                current = (int) materials_.size();
                materials_.push_back(m);
                name2idx[nm] = current;
            }
        }
        else if(current >= 0 && t0 == "Kd" && tok.size() >= 4){
            materials_[current].Kd = glm::vec3(std::stof(tok[1]),
                std::stof(tok[2]),
                std::stof(tok[3]));
        }
        else if(current >= 0 && (t0 == "map_Kd" || t0 == "map_kd") && tok.size() >= 2){
            // 忽略 -o/-s/... 取最後一個不以 '-' 開頭的 token 當路徑
            std::string texrel;
            for(int i = (int) tok.size() - 1; i >= 1; --i){
                if(!tok[i].empty() && tok[i][0] != '-'){ texrel = tok[i]; break; }
            }
            if(texrel.empty()) continue;
            std::string texpath = (fs::path(base_dir) / fs::path(texrel)).string();
            materials_[current].map_Kd_path = texpath;
            stbi_set_flip_vertically_on_load(1);
            materials_[current].tex = load_texture(texpath);
        }
    }

    // 若整份 MTL 只有一張貼圖，補到其他材質上（可留）
    int texCount = 0; std::string texPath;
    for(auto &m : materials_) if(m.tex){ ++texCount, texPath = m.map_Kd_path; }
    if(texCount == 1 && !texPath.empty()){
        for(auto &m : materials_){
            if(!m.tex){
                m.map_Kd_path = texPath;
                m.tex = load_texture(texPath);
            }
        }
    }
    return true;
}

GLuint Mesh::load_texture(const std::string &path){
    int w, h, n;
    stbi_uc *data = stbi_load(path.c_str(), &w, &h, &n, 4);
    if(!data){
        std::cerr << "[Mesh] Failed to load texture: " << path << "\n";
        return 0;
    }
    std::cerr << "[Mesh] Texture OK: " << path << " (" << w << "x" << h << ")\n";
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(data);
    return tex;
}

// ------------------------- small utils -------------------------
std::string Mesh::read_text_file(const std::string &path){
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}
std::string Mesh::dir_of(const std::string &path){
    fs::path p(path);
    auto d = p.parent_path();
    return d.empty() ? string(".") : d.string();
}
std::string Mesh::trim(const std::string &s){
    size_t a = 0, b = s.size();
    while(a < b && std::isspace((unsigned char) s[a])) ++a;
    while(b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}
std::vector<std::string> Mesh::split_ws(const std::string &s){
    std::vector<string> out;
    std::istringstream in(s);
    string t;
    while(in >> t) out.push_back(t);
    return out;
}

// ------------------------- in-memory builders -------------------------

bool Mesh::load_from_interleaved(const std::vector<float> &interleaved,
    const std::vector<unsigned int> &indices,
    const glm::vec3 &solidKd,
    GLuint texture){
    clear();
    if(interleaved.empty() || indices.empty()){
        std::cerr << "[Mesh] load_from_interleaved: empty data\n";
        return false;
    }
    if((interleaved.size() % 8 != 0) && (interleaved.size() % 11 != 0)){
        std::cerr << "[Mesh] load_from_interleaved: size not multiple of 8 or 11\n";
        return false;
    }
    vertices_ = interleaved;
    indices_ = indices;

    stride_floats_ = ((vertices_.size() % 11) == 0 ? 11 : 8);
    has_color_ = (stride_floats_ == 11);

    ranges_.clear();
    DrawRange r; r.material_id = 0; r.index_offset = 0; r.index_count = static_cast<GLsizei>(indices_.size());
    ranges_.push_back(r);

    materials_.clear();
    Material m; m.name = "inmem"; m.Kd = solidKd; m.tex = texture;
    materials_.push_back(m);

    create_gl_buffers();
    ok_ = true;
    return true;
}

bool Mesh::load_from_memory(const std::vector<glm::vec3> &positions,
    const std::vector<glm::vec3> *normals,
    const std::vector<glm::vec2> *uvs,
    const std::vector<unsigned int> &indices,
    const glm::vec3 &solidKd,
    GLuint texture){
    // wrapper（無頂點色）
    return load_from_memory(positions, normals, uvs,
        (const std::vector<glm::vec3>*)nullptr,
        indices, solidKd, texture);
}

bool Mesh::load_from_memory(const std::vector<glm::vec3> &positions,
    const std::vector<glm::vec3> *normals,
    const std::vector<glm::vec2> *uvs,
    const std::vector<glm::vec3> *colors,
    const std::vector<unsigned int> &indices,
    const glm::vec3 &solidKd,
    GLuint texture){
    clear();
    if(positions.empty() || indices.empty()){
        std::cerr << "[Mesh] load_from_memory(colors): positions/indices empty\n";
        return false;
    }
    bool hasN = (normals && normals->size() == positions.size());
    bool hasT = (uvs && uvs->size() == positions.size());
    bool hasC = (colors && colors->size() == positions.size());

    stride_floats_ = hasC ? 11 : 8;
    has_color_ = hasC;
    vertices_.reserve(positions.size() * stride_floats_);
    for(size_t i = 0; i < positions.size(); ++i){
        const glm::vec3 &P = positions[i];
        glm::vec3 N = hasN ? (*normals)[i] : glm::vec3(0, 0, 1);
        glm::vec2 T = hasT ? (*uvs)[i] : glm::vec2(0, 0);
        vertices_.push_back(P.x); vertices_.push_back(P.y); vertices_.push_back(P.z);
        vertices_.push_back(N.x); vertices_.push_back(N.y); vertices_.push_back(N.z);
        vertices_.push_back(T.x); vertices_.push_back(T.y);
        if(hasC){
            const glm::vec3 &C = (*colors)[i];
            vertices_.push_back(C.r); vertices_.push_back(C.g); vertices_.push_back(C.b);
        }
    }
    indices_ = indices;

    ranges_.clear();
    DrawRange r; r.material_id = 0; r.index_offset = 0; r.index_count = static_cast<GLsizei>(indices_.size());
    ranges_.push_back(r);

    materials_.clear();
    Material m; m.name = "inmem"; m.Kd = solidKd; m.tex = texture;
    materials_.push_back(m);

    create_gl_buffers();
    ok_ = true;
    return true;
}

void Mesh::set_material(int i, const glm::vec3 &Kd, GLuint texture){
    if(i < 0) return;
    if(i >= (int) materials_.size()){
        materials_.resize(i + 1);
    }
    materials_[i].Kd = Kd;
    materials_[i].tex = texture;
}

