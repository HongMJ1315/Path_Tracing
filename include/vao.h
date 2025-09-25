#include <vector>

#include "GLinclude.h"

class VertexArray{
public:
    VertexArray();
    ~VertexArray();

    // Bind/unbind VAO
    void bind() const;
    void unbind() const;

    // Upload entire vertex buffer (initialization)
    void setVertexBuffer(const std::vector<float> &data, GLenum usage = GL_STATIC_DRAW);

    // Update existing vertex buffer content (subdata)
    void updateVertexBuffer(const std::vector<float> &data, GLintptr offset = 0);

    // Upload index buffer
    void setIndexBuffer(const std::vector<unsigned int> &indices, GLenum usage = GL_STATIC_DRAW);

    // Define vertex attribute layout
    void setAttribPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);

    // Accessor for VBO if needed elsewhere
    GLuint getVBO() const;

private:
    GLuint VAO;
    GLuint VBO;
    GLuint EBO;
};
    