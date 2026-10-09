#version 440

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
};

void main()
{
    fragColor = vColor;
}
