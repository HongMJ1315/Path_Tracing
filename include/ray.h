// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

enum RayType{
    EYE, LIGHT
};

class Ray{
public:
    glm::vec3 point;
    glm::vec3 vec;
    float refract;
    RayType type;  
    Ray(){}
    Ray(glm::vec3 point, glm::vec3 vec, float reftact, RayType type){
        this->point = point;
        this->vec = vec;
        this->refract = refract;
        this->type = type;
    }
};