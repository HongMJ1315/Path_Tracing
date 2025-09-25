// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class Ray{
public:
    glm::vec3 point;
    glm::vec3 vec;
    Ray(){}
    Ray(glm::vec3 point, glm::vec3 vec){
        this->point = point;
        this->vec = vec;
    }
};