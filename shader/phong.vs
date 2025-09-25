#version 410 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in vec3 aColor;

out vec3 vPosW;
out vec3 vNrmW;
out vec2 vUV;
out vec3 vColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

void main(){
    vec4 Pw = uModel * vec4(aPos, 1.0);
    vPosW = Pw.xyz;
    mat3 Nmat = mat3(transpose(inverse(uModel)));
    vNrmW = normalize(Nmat * aNormal);
    vUV   = aUV;
    vColor= aColor;
    gl_Position = uProj * uView * Pw;
}
