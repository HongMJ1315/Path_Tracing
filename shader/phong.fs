#version 410 core
in vec3 vPosW;
in vec3 vNrmW;
in vec2 vUV;
in vec3 vColor;
layout(location=0) out vec4 FragColor;

struct Light {
    int   type;        // 0:Directional, 1:Point, 2:Spot
    vec3  color;
    float intensity;
    vec3  position;
    float range;
    vec3  direction;   // points toward scene
    float innerCutoff; // cos(inner)
    float outerCutoff; // cos(outer)
};

uniform int   uNumLights;
uniform Light uLights[8];
uniform vec3  uCameraPos;

uniform sampler2D uTex0;
uniform bool  uHasTex;
uniform bool  uUseVertexColor;
uniform vec3  uKd;
uniform vec3  uKs;
uniform float uShininess;
uniform float uAmbient;

float attenPoint(float dist, float range){
    float x = clamp(dist / max(range, 1e-4), 0.0, 1.0);
    return 1.0 / (1.0 + 4.0*x + 8.0*x*x);
}
float spotFactor(vec3 Ldir_norm, vec3 spotDir_norm, float innerCos, float outerCos){
    float cosAng = dot(-spotDir_norm, Ldir_norm);
    return clamp((cosAng - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
}

vec3 baseColor(){
    vec3 kd = uKd;
    if(uHasTex) kd = texture(uTex0, vUV).rgb;
    else if(uUseVertexColor) kd = vColor;
    return kd;
}

void main(){
    vec3 N = normalize(vNrmW);
    vec3 V = normalize(uCameraPos - vPosW);
    vec3 kd = baseColor();
    vec3 ks = uKs;
    float shin = max(uShininess, 1.0);

    vec3 color = kd * uAmbient;

    for(int i=0; i<uNumLights; ++i){
        Light L = uLights[i];
        vec3 Ldir;
        float att = 1.0;
        float spot = 1.0;

        if(L.type == 0){
            Ldir = normalize(-L.direction);
        }else{
            vec3 toL = L.position - vPosW;
            float dist = length(toL);
            Ldir = toL / max(dist, 1e-4);
            att = 1.0 / (1.0 + 4.0*(dist/L.range) + 8.0*pow(dist/L.range,2.0));
            // 若是 Spotlight 再算 spot（你原來的邏輯）
            if(L.type == 2){
                float innerCos = L.innerCutoff, outerCos = L.outerCutoff;
                float cosAng = dot(normalize(L.direction), normalize(vPosW - L.position));
                spot = clamp((cosAng - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
            }
        }

        float NdotL = max(dot(N, Ldir), 0.0);
        vec3 diffuse = kd * NdotL;

        // ★ 修正點：高光只在 N·L > 0 時出現
        float specMask = step(0.0, NdotL);
        vec3 H = normalize(Ldir + V);
        float NdotH = max(dot(N, H), 0.0);
        vec3 specular = ks * pow(NdotH, shin) * specMask;

        vec3 Lrgb = L.color * L.intensity;
        color += (diffuse + specular) * Lrgb * att * spot;
    }

    FragColor = vec4(color, 1.0);
}