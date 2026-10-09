#version 440
// Crosshair marks drawn as world geometry in the surface's pass, depth-tested
// so the brain hides them. Layout must match viewer/surface3d.py:crosshair_geometry.

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 rgba;

layout(location = 0) out vec4 vColor;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
};

void main()
{
    vColor = rgba;
    gl_Position = mvp * vec4(pos, 1.0);
}
