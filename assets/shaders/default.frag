
#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_model_to_clip;
    mat4 mtx_model_to_view_norm;
} ubo;
layout (binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846264338327950288;

void main() {
    vec3 normTS = fragNormal;
    vec3 normVS = (ubo.mtx_model_to_view_norm * vec4(fragNormal, 1.0)).xyz;
    normVS = normalize(normVS);
    vec3 d = vec3(0, 0, -1);
    vec3 r = d - (2.0 * dot(d, normVS) * normVS);
    r.y *= -1.0;
    vec2 uv = vec2(atan(r.x, r.z) / 2.0, asin(r.y)) / PI + 0.5;
    outColor = texture(texSampler, uv);
}
