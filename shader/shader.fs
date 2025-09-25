#version 410 core
in vec2 vUV;
layout(location=0) out vec4 FragColor;
uniform sampler2D uTex0;
uniform bool uHasTex;
uniform vec3 uKd;
void main(){
  vec3 kd = uHasTex ? texture(uTex0, vUV).rgb : uKd;
  FragColor = vec4(kd, 1.0);
}
