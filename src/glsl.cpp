#include "glsl.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

int Shader::num_shaders = 0;

Shader::Shader(const char *vertex_path, const char *fragment_path){
    ID = glCreateProgram();

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);

    char *vert_src = read_source_codes(const_cast<char *>(vertex_path));
    char *frag_src = read_source_codes(const_cast<char *>(fragment_path));

    std::cerr << "Vertex Shader Source:\n" << vert_src << "\n";
    glShaderSource(vert, 1, &vert_src, NULL);
    glCompileShader(vert);
    print_shader_info_log(vert);

    std::cerr << "Fragment Shader Source:\n" << frag_src << "\n";
    glShaderSource(frag, 1, &frag_src, NULL);
    glCompileShader(frag);
    print_shader_info_log(frag);

    free(vert_src);
    free(frag_src);

    glAttachShader(ID, vert);
    glAttachShader(ID, frag);
    glLinkProgram(ID);
    print_prog_info_log(ID);

    glDeleteShader(vert);
    glDeleteShader(frag);

    num_shaders++;
    std::cout << "Created Shader program ID = " << ID
        << "  (total shaders: " << num_shaders << ")\n";
}

Shader::~Shader(){
    glDeleteProgram(ID);
    num_shaders--;
}

void Shader::set_bool(const std::string &name, bool value) const{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int) value);
}
void Shader::set_int(const std::string &name, int value) const{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::set_float(const std::string &name, float value) const{
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::set_vec2(const std::string &name, const glm::vec2 &v) const{
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(v));
}
void Shader::set_vec3(const std::string &name, const glm::vec3 &v) const{
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(v));
}
void Shader::set_mat4(const std::string &name, const glm::mat4 &m) const{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()),
        1, GL_FALSE, glm::value_ptr(m));
}
