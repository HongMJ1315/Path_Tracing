#include "GLinclude.h"


GLuint tex = 0;
GLuint vao, vbo;
Shader *shader;
GLuint prog = 0;

void gen_texture(int w, int h){
    // --- 放在 main() 開頭初始化完成後、進入渲染前 ---
// 解析度
    const int W = w * 2;
    const int H = h;

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

GLuint get_vao(){
    return vao;
}

GLuint get_shader(){
    return prog;
}

GLuint get_texture(){
    return tex;
}