
#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846264338327950288;

void main() {
    vec3 r = fragNormal;
    float u = (atan(r.x/r.z) + PI) / (3.0 * PI);
    float v = (asin(r.y) + 0.5 * PI) / PI;
    outColor = texture(texSampler, vec2(u, v));
}
