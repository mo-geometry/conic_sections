"""Built-in shader sources for the rendering pipeline."""

# --- Blinn-Phong shading with optional texture ---

VERTEX_SHADER = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord;

out vec3 frag_pos;
out vec3 frag_normal;
out vec2 frag_texcoord;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_texcoord = in_texcoord;
    gl_Position = projection * view * world_pos;
}
"""

FRAGMENT_SHADER = """
#version 330 core

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 object_color;
uniform vec3 view_pos;
uniform bool use_texture;
uniform sampler2D tex;

in vec3 frag_pos;
in vec3 frag_normal;
in vec2 frag_texcoord;

out vec4 frag_color;

void main() {
    // Determine surface colour from texture or uniform
    vec3 surface_color;
    if (use_texture) {
        surface_color = texture(tex, frag_texcoord).rgb;
    } else {
        surface_color = object_color;
    }

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

    vec3 result = (ambient + diffuse + specular) * surface_color;
    frag_color = vec4(result, 1.0);
}
"""

# --- Grid shader for ground plane ---

GRID_VERTEX_SHADER = """
#version 330 core

uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_color;

out vec3 frag_color_v;

void main() {
    frag_color_v = in_color;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

GRID_FRAGMENT_SHADER = """
#version 330 core

in vec3 frag_color_v;
out vec4 frag_color;

void main() {
    frag_color = vec4(frag_color_v, 1.0);
}
"""
