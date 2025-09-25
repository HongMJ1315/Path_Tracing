#version 410 core
in vec2 vUV;
layout(location=0) out vec4 FragColor;

struct Light {
    int   type;         // 0: directional, 1: point
    vec3  color;
    float intensity;
    vec3  position;
    float range;
    vec3  direction;
    float _pad0;
    float _pad1;
};

uniform int   uNumLights;
uniform Light uLights[8];
uniform vec3  uCameraPos;

// for camera ray
uniform mat4  uInvViewProj;

// tetra (object B) in world
uniform vec3  uTetPos[4];        // p0..p3
uniform vec3  uTetCol[4];        // c0..c3

// gizmo spheres (lights)
uniform vec3  uSunProxyPos;      // 方向光的代理位置（相機前方某點）
uniform vec3  uBulbPos;
uniform float uSphereR;

// material/params (re-use your phong params)
uniform vec3  uKs;
uniform float uShininess;
uniform float uAmbient;

// -----------------------------------------------
// utils
bool rayTriangle(vec3 ro, vec3 rd, vec3 a, vec3 b, vec3 c, out float t, out vec3 n, out vec3 bary){
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 p  = cross(rd, ac);
    float det = dot(ab, p);
    if (abs(det) < 1e-7) return false;
    float invDet = 1.0 / det;
    vec3 tvec = ro - a;
    float u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(tvec, ab);
    float v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;
    float tt = dot(ac, q) * invDet;
    if (tt <= 1e-4) return false;
    t = tt;

    // 以三角形重心判斷外向（物體在原點附近的前提）
    n = normalize(cross(ab, ac));  // ★ 只用繞向
    bary = vec3(1.0 - u - v, u, v);
    return true;
}

bool raySphere(vec3 ro, vec3 rd, vec3 center, float r, out float t, out vec3 n){
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - r*r;
    float h = b*b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;
    float tt = t0 > 1e-4 ? t0 : (t1 > 1e-4 ? t1 : -1.0);
    if (tt <= 0.0) return false;
    t = tt;
    vec3 hit = ro + rd * t;
    n = normalize(hit - center);
    return true;
}

float attenPoint(float dist, float range){
    float x = clamp(dist / max(range, 1e-4), 0.0, 1.0);
    return 1.0 / (1.0 + 4.0*x + 8.0*x*x);
}

vec3 shadePhong(vec3 P, vec3 N, vec3 base){
    vec3 V = normalize(uCameraPos - P);
    vec3 color = base * uAmbient;

    for (int i=0;i<uNumLights;++i){
        Light L = uLights[i];
        vec3 Ldir;
        float att = 1.0;
        if (L.type == 0){
            Ldir = normalize(-L.direction);
        }else{
            vec3 toL = L.position - P;
            float dist = length(toL);
            Ldir = toL / max(dist, 1e-4);
            att  = attenPoint(dist, L.range);
        }
        float NdotL = max(dot(N, Ldir), 0.0);
        vec3 diffuse = base * NdotL;

        float specMask = step(0.0, NdotL);
        vec3 H = normalize(Ldir + V);
        float NdotH = max(dot(N, H), 0.0);
        vec3 specular = uKs * pow(NdotH, max(uShininess,1.0)) * specMask;

        vec3 Lrgb = L.color * L.intensity;
        color += (diffuse + specular) * Lrgb * att;
    }
    return color;
}

// -----------------------------------------------

void main(){
    // 反投影：NDC -> world ray
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 pNear = uInvViewProj * vec4(ndc, 0.0, 1.0);
    vec4 pFar  = uInvViewProj * vec4(ndc, 1.0, 1.0);
    vec3 ro = uCameraPos;
    vec3 rd = normalize( (pFar.xyz/pFar.w) - (pNear.xyz/pNear.w) );

    float tBest = 1e30;
    vec3  Nbest = vec3(0);
    vec3  Cbest = vec3(0);

    // --- 1) tetra: 4 faces, interpolate vertex color by barycentric ---
    // faces: (0,1,2), (0,3,1), (0,2,3), (1,3,2)  // 與你的建立順序一致
    ivec3 faces[4] = ivec3[4]( ivec3(0,1,2), ivec3(0,3,1), ivec3(0,2,3), ivec3(1,3,2) );
    for (int f=0; f<4; ++f){
        vec3 a = uTetPos[faces[f].x];
        vec3 b = uTetPos[faces[f].y];
        vec3 c = uTetPos[faces[f].z];
        float t; vec3 n, bary;
        if (rayTriangle(ro, rd, a, b, c, t, n, bary)){
            if (t < tBest){
                tBest = t; Nbest = n;
                vec3 ca = uTetCol[faces[f].x];
                vec3 cb = uTetCol[faces[f].y];
                vec3 cc = uTetCol[faces[f].z];
                Cbest = ca*bary.x + cb*bary.y + cc*bary.z;
            }
        }
    }

    // --- 2) spheres at lights ---
    {
        float t; vec3 n;
        if (raySphere(ro, rd, uSunProxyPos, uSphereR, t, n) && t < tBest){
            tBest = t; Nbest = n; Cbest = vec3(1.0,1.0,0.0); // #ffff00
        }
        if (raySphere(ro, rd, uBulbPos, uSphereR, t, n) && t < tBest){
            tBest = t; Nbest = n; Cbest = vec3(0.0,1.0,1.0); // #00ffff
        }
    }

    if (tBest < 1e20){
        vec3 P = ro + rd * tBest;
        vec3 N = normalize(Nbest);
        if (dot(N, rd) > 0.0) N = -N;   // ★ 這行很關鍵
        vec3 color = shadePhong(P, N, Cbest);
        FragColor = vec4(color, 1.0);
    }else{
        discard; // ★ 混合光柵時不要覆蓋背景
    }

}
