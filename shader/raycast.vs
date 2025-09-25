#version 410 core
// Fullscreen big triangle
const vec2 vtx[3] = vec2[3](
    vec2(-1.0,-1.0),
    vec2( 3.0,-1.0),
    vec2(-1.0, 3.0)
);
out vec2 vUV;
void main(){
    vUV = vtx[gl_VertexID]*0.5 + 0.5;
    gl_Position = vec4(vtx[gl_VertexID], 0.0, 1.0);
}
