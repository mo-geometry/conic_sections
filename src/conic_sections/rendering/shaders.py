"""Built-in shader sources for the rendering pipeline."""

# --- Basic Phong shading ---

VERTEX_SHADER = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_normal;

out vec3 frag_pos;
out vec3 frag_normal;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    gl_Position = projection * view * world_pos;
}
"""

FRAGMENT_SHADER = """
#version 330 core

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 object_color;
uniform vec3 view_pos;

in vec3 frag_pos;
in vec3 frag_normal;

out vec4 frag_color;

void main() {
    // Ambient
    float ambient_strength = 0.15;
    vec3 ambient = ambient_strength * light_color;

    // Diffuse
    vec3 norm = normalize(frag_normal);
    vec3 light_dir = normalize(light_pos - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    // Specular (Blinn-Phong)
    float specular_strength = 0.5;
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0);
    vec3 specular = specular_strength * spec * light_color;

    vec3 result = (ambient + diffuse + specular) * object_color;
    frag_color = vec4(result, 1.0);
}
"""
