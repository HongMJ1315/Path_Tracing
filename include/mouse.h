#pragma once
#include "GLinclude.h"

// Camera control state
extern float yaw;      // rotation around Y axis (left/right)
extern float pitch;    // rotation around X axis (up/down)
extern float radius;   // distance from origin (zoom)

// Internal state
extern bool rotating;
extern double lastX, lastY;

// Callbacks
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

