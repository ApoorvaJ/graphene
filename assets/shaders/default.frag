#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_model_to_clip;
    mat4 mtx_model_to_view;
    mat4 mtx_model_to_view_norm;
    float elapsed_seconds;
} ubo;
layout (binding = 1) uniform sampler2D tex_sampler;
layout(location = 0) in vec3 frag_norm_vs;
layout(location = 1) in vec3 frag_pos_vs;
layout(location = 0) out vec4 out_color;

const float PI = 3.14159265358979323846264338327950288;

void main() {
    vec3 n = normalize(frag_norm_vs);

    float roughness = cos(ubo.elapsed_seconds) * 0.5 + 0.5;
    // float roughness = 1.0;
    vec3 v = vec3(0, 0, -1);
    // vec3 v = -normalize(frag_pos_vs);

    const vec3 lights[2] = vec3[](
        vec3(0, -1, 0),
        vec3(1, 1, 0)
    );
    const float intensities[2] = float[] (
        1.0, 0.2
    );
    vec3 final = vec3(0, 0, 0);

    for (int i = 0; i < 2; i++) {

        vec3 h = normalize(v + lights[i]);

        float n_dot_v = abs(dot(n, v)) + 1e-5;
        float n_dot_l = clamp(dot(n, lights[i]), 0.0, 1.0);
        float n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        float l_dot_h = clamp(dot(lights[i], h), 0.0, 1.0);

        float d_ggx;
        {
            float a2 = roughness * roughness;
            float f = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
            d_ggx = a2 / (PI * f * f);
        }

        float v_ggx;
        {
            float a2 = roughness * roughness;
            float ggxl = n_dot_v * sqrt((-n_dot_l * a2 + n_dot_l) * n_dot_l + a2);
            float ggxv = n_dot_l * sqrt((-n_dot_v * a2 + n_dot_v) * n_dot_v + a2);
            v_ggx = 0.5 / (ggxv + ggxl);
        }

        vec3 f_schlick;
        {
            vec3 f0 = vec3(0.97, 0.74, 0.62);
            float f = pow(1.0 - l_dot_h, 5.0);
            f_schlick = f + f0 * (1.0 - f);
        }

        final += intensities[i] * n_dot_l * d_ggx * v_ggx * f_schlick;
    }

    out_color = vec4(final, 1.0);
}
