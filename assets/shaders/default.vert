
#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBuffer {
    mat4 mtx_world_to_clip;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {

    gl_Position = ubo.mtx_world_to_clip * vec4(inPosition, 1.0);
    fragColor = inNormal;
}
