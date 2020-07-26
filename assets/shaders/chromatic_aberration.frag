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

float noise1d(float n){
    return fract(cos(n*89.42)*343.42);
}
vec3 noise2d(vec2 co){
  float r = fract(sin(dot(co.xy ,vec2(1.0,73))) * 43758.5453);
  float g = noise1d(r);
  float b = noise1d(g);
  return vec3(r, g, b);
}

void main() {
    vec2 viewport_size = vec2(ubo.viewport_w, ubo.viewport_h);
    vec2 uv = gl_FragCoord.xy / viewport_size;
    float t = ubo.elapsed_seconds;

    float noise = fract(noise1d(t));
    
    vec2 uv_r = uv + vec2(1, 1) * 0.012;
    vec2 uv_g = uv + vec2(1, 0.2) * 0.008;

    out_color.r = texture(tex_sampler, uv_r).r;
    out_color.g = texture(tex_sampler, uv_g).g;
    out_color.b = texture(tex_sampler, uv).b;

}
