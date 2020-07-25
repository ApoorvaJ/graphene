#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_obj_to_clip;
    mat4 mtx_norm_obj_to_world;
    float elapsed_seconds;
} ubo;
layout (binding = 1) uniform sampler2D tex_sampler;
layout(location = 0) in vec3 frag_norm_world;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(1.0, 1.0, 0.0, 1.0);
    return;
}
