#include "light.h"
#include <sstream>

static std::string u(const std::string &base, int idx, const char *field){
    std::ostringstream os;
    os << base << "[" << idx << "]." << field;
    return os.str();
}

void Light::apply(Shader &shader, int index, const std::string &baseName) const{
    shader.set_int(u(baseName, index, "type"), static_cast<int>(type));
    shader.set_vec3(u(baseName, index, "color"), color);
    shader.set_float(u(baseName, index, "intensity"), intensity);

    shader.set_vec3(u(baseName, index, "position"), position);
    shader.set_float(u(baseName, index, "range"), range);

    shader.set_vec3(u(baseName, index, "direction"), glm::normalize(direction));
    shader.set_float(u(baseName, index, "innerCutoff"), glm::cos(innerCutoff)); // store cos for cheaper shader
    shader.set_float(u(baseName, index, "outerCutoff"), glm::cos(outerCutoff));
}