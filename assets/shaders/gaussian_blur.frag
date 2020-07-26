#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_obj_to_clip;
    mat4 mtx_norm_obj_to_world;
    float elapsed_seconds;
    float viewport_w;
    float viewport_h;
} ubo;
layout (binding = 1) uniform sampler2D tex_sampler;
layout(location = 0) in vec3 frag_norm_world;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 viewport_size = vec2(ubo.viewport_w, ubo.viewport_h);
    vec2 uv = gl_FragCoord.xy / viewport_size;
    out_color = texture(tex_sampler, uv);
}
