#version 450

// TODO: Remove all of these, once this part is dynamic in the graphics pipeline
layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_obj_to_clip;
    mat4 mtx_norm_obj_to_world;
    float elapsed_seconds;
    float viewport_w;
    float viewport_h;
} ubo;
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 0) out vec3 frag_norm_world;
//

out gl_PerVertex {
    vec4 gl_Position;
};

vec2 positions[3] = vec2[](
    vec2(-1, -1),
    vec2(-1, 3),
    vec2(3, -1)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
}
