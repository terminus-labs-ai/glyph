shader_type canvas_item;
render_mode blend_mix, unshaded;

// Noise texture with albedo hint
uniform sampler2D noise_tex : hint_albedo;

// Animation speed parameter
uniform float speed : hint_range(0.0, 10.0) = 1.0;

void vertex() { VERTEX.x += sin(TIME) * speed; }

void fragment() { COLOR = texture(noise_tex, UV); }
