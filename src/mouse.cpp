#include "mouse.h"
#include <iostream>

// Initialize defaults
float yaw = 0.0f;
float pitch = 0.0f;
float radius = 5.0f;
bool  rotating = false;
double lastX = 0.0, lastY = 0.0;

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        if(action == GLFW_PRESS){
            rotating = true;
            glfwGetCursorPos(window, &lastX, &lastY);
        }
        else if(action == GLFW_RELEASE){
            rotating = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos){
    if(!rotating) return;
    float sensitivity = 0.005f;
    double dx = xpos - lastX;
    double dy = ypos - lastY;
    lastX = xpos;
    lastY = ypos;

    yaw += -float(dx * sensitivity);
    pitch += float(dy * sensitivity);
    // clamp pitch to avoid gimbal lock
    if(pitch > 1.5f) pitch = 1.5f;
    if(pitch < -1.5f) pitch = -1.5f;
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset){
    float zoomSpeed = 0.5f;
    radius -= float(yoffset * zoomSpeed);
    if(radius < 1.0f) radius = 1.0f;
    if(radius > 20.0f) radius = 20.0f;
}

