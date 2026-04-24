#version 450

// Camera uniform block with layout qualifier
layout(std140, binding = 0) uniform CameraBlock {
    mat4 viewProjection;
    vec3 cameraPos;
};

// Sampler resource for albedo texture
uniform sampler2D albedoMap;

// Input from vertex shader
layout(location = 0) in vec2 fragTexCoord;

// Output color
layout(location = 0) out vec4 outColor;

/// Helper to compute directional lighting
vec3 computeLighting(vec3 normal, vec3 lightDir) { return max(dot(normal, lightDir), 0.0) * vec3(1.0); }

void main() { outColor = vec4(texture(albedoMap, fragTexCoord).rgb, 1.0); }
