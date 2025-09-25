#ifndef LIGHT_H
#define LIGHT_H

#include "GLinclude.h"
#include "glsl.h"
#include <string>

class Light{
public:
    enum class Type : int{ Directional = 0, Point = 1, Spot = 2 };

    // Parameters
    Type      type = Type::Directional;
    glm::vec3 color = glm::vec3(1.0f);
    float     intensity = 1.0f;

    // For Point / Spot
    glm::vec3 position = glm::vec3(3.0f, 3.0f, 3.0f);
    float     range = 10.0f;   // attenuation radius

    // For Directional / Spot
    glm::vec3 direction = glm::vec3(-0.5f, -1.0f, -0.2f);  // pointing *toward* scene
    float     innerCutoff = glm::radians(15.0f);             // radians
    float     outerCutoff = glm::radians(25.0f);             // radians

    Light() = default;
    explicit Light(Type t) : type(t){}

    // Upload to shader as uLights[index].*
    void apply(Shader &shader, int index, const std::string &baseName = "uLights") const;
};

#endif // LIGHT_H