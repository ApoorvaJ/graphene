#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_model_to_clip;
    mat4 mtx_model_to_view;
    mat4 mtx_model_to_view_norm;
    float elapsed_seconds;
} ubo;
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 0) out vec3 frag_norm_vs;
layout(location = 1) out vec3 frag_pos_vs;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.mtx_model_to_clip * vec4(in_pos, 1.0);
    frag_norm_vs = (ubo.mtx_model_to_view_norm * vec4(in_norm, 1.0)).xyz;
    frag_pos_vs = (ubo.mtx_model_to_view * vec4(in_pos, 1.0)).xyz;
}
