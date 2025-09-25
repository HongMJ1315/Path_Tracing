// vao_vbo.h
#pragma once


// vao_vbo.cpp
#include "vao.h"

VertexArray::VertexArray(){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
}

VertexArray::~VertexArray(){
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void VertexArray::bind() const{
    glBindVertexArray(VAO);
}

void VertexArray::unbind() const{
    glBindVertexArray(0);
}

void VertexArray::setVertexBuffer(const std::vector<float> &data, GLenum usage){
    bind();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), usage);
}

void VertexArray::updateVertexBuffer(const std::vector<float> &data, GLintptr offset){
    bind();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, offset, data.size() * sizeof(float), data.data());
}

void VertexArray::setIndexBuffer(const std::vector<unsigned int> &indices, GLenum usage){
    bind();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), usage);
}

void VertexArray::setAttribPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer){
    bind();
    glEnableVertexAttribArray(index);
    glVertexAttribPointer(index, size, type, GL_FALSE, stride, pointer);
}

GLuint VertexArray::getVBO() const{
    return VBO;
}
