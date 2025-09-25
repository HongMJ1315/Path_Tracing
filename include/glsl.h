#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "GLinclude.h"  // 包含 GLFW, glad, GL 类型定义等

class Shader{
public:
    Shader(const char *vertexPath, const char *fragmentPath);
    ~Shader();

    void use() const{ glUseProgram(ID); }

    void set_bool(const std::string &name, bool value)   const;
    void set_int(const std::string &name, int value)    const;
    void set_float(const std::string &name, float value)  const;
    void set_vec2(const std::string &name, const glm::vec2 &v) const;
    void set_vec3(const std::string &name, const glm::vec3 &v) const;
    void set_mat4(const std::string &name, const glm::mat4 &m) const;

    GLuint ID;

private:
    static int num_shaders;
    char *read_source_codes(const char *filename) const{
        FILE *fptr = fopen(filename, "r");
        if(!fptr){
            fprintf(stderr, " Source code file: %s doesn't exist.\n", filename);
            exit(EXIT_FAILURE);
        }
        fseek(fptr, 0, SEEK_END);
        unsigned int size = ftell(fptr);
        rewind(fptr);

        char *buffer = (char *) malloc(size + 1);
        size = fread(buffer, sizeof(char), size, fptr);
        buffer[size] = '\0';
        fclose(fptr);
        return buffer;
    }

    void print_shader_info_log(GLuint obj) const{
        int status; char infoLog[1024];
        glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
        if(status == GL_FALSE){
            glGetShaderInfoLog(obj, 1024, NULL, infoLog);
            fprintf(stderr, "Shader compile error: %s\n", infoLog);
            exit(EXIT_FAILURE);
        }
        std::cerr << "Shader compiling is successful.\n";
    }

    void print_prog_info_log(GLuint obj) const{
        int status; char infoLog[1024];
        glGetProgramiv(obj, GL_LINK_STATUS, &status);
        if(status == GL_FALSE){
            glGetProgramInfoLog(obj, 1024, NULL, infoLog);
            fprintf(stderr, "Shader link error: %s\n", infoLog);
            exit(EXIT_FAILURE);
        }
        std::cerr << "Shader linking is successful.\n";
    }
};
