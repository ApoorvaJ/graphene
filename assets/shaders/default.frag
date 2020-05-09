#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_model_to_clip;
    mat4 mtx_model_to_view_norm;
} ubo;
layout (binding = 1) uniform sampler2D tex_sampler;
layout(location = 0) in vec3 frag_norm;
layout(location = 0) out vec4 out_color;

const float PI = 3.14159265358979323846264338327950288;

void main() {
    vec3 norm_vs = (ubo.mtx_model_to_view_norm * vec4(frag_norm, 1.0)).xyz;
    // Removing this makes things look faceted. Why? Losing precision?
    norm_vs = normalize(norm_vs);
    vec3 d = vec3(0, 0, -1);
    vec3 r = d - (2.0 * dot(d, norm_vs) * norm_vs);
    vec2 uv = vec2(atan(r.x, r.z) / 2.0, asin(r.y)) / PI + 0.5;
    out_color = texture(tex_sampler, uv);
}
