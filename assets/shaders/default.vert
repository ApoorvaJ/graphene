#version 450

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_obj_to_clip;
    mat4 mtx_norm_obj_to_world;
    float elapsed_seconds;
} ubo;
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 0) out vec3 frag_norm_world;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.mtx_obj_to_clip * vec4(in_pos, 1.0);
    frag_norm_world = (ubo.mtx_norm_obj_to_world * vec4(in_norm, 1.0)).xyz;
}
