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

const float PI = 3.14159265358979323846264338327950288;

vec3 f_schlick(vec3 f0, float f90, float u) {
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

void main() {
    vec3 n = normalize(frag_norm_world);
    out_color = vec4(n.xyz, 1.0);
    return;

    // float roughness = cos(ubo.elapsed_seconds) * 0.5 + 0.5;
    float roughness = 1.0;
    roughness = clamp(roughness * roughness, 1e-5, 1.0);
    vec3 v = vec3(0, 0, -1);

    const vec3 lights[2] = vec3[](
        vec3(0, -1, 0),
        vec3(1, 1, 0)
    );
    const float intensities[2] = float[] (
        1.0, 0.2
    );
    vec3 final = vec3(0, 0, 0);

    for (int i = 0; i < 1; i++) {

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

        vec3 fresnel_specular;
        {
            vec3 f0 = vec3(0.04, 0.04, 0.04);
            float f90 = 1.0;
            fresnel_specular = f_schlick(f0, f90, l_dot_h);
        }

        float diffuse_disney;
        {
            float energy_bias = mix(0, 0.5, roughness);
            float energy_factor = mix(1.0, 1.0 / 1.51, roughness);
            float fd90 = energy_bias + 2.0 * l_dot_h * l_dot_h * roughness;
            vec3 f0 = vec3(1.0, 1.0, 1.0);
            float light_scatter = f_schlick(f0, fd90, n_dot_l).r;
            float view_scatter = f_schlick(f0, fd90, n_dot_v).r;
            diffuse_disney = light_scatter * view_scatter * energy_factor;
        }

        vec3 f_r = d_ggx * v_ggx * fresnel_specular;
        vec3 diffuse_color = vec3(1.0, 1.0, 1.0);
        vec3 f_d = diffuse_disney * diffuse_color;

        final += intensities[i] * n_dot_l * (f_r + f_d);
    }

    out_color = vec4(final, 1.0);
}
