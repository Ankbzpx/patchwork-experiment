#version 330
layout(location = 0) out vec4 fragColor;

in vec2 uv;
uniform sampler2D texture1;

void main() {
    vec3 lightColor = vec3(0.8510, 0.9137, 0.8118);
    vec3 darkColor = vec3(0.5882, 0.6549, 0.5529);
    float d = texture(texture1, uv).r;

    vec3 col = (d > 0.0) ? lightColor : darkColor;
    col *= 1.0 - exp(-6.0 * abs(d));
    col *= 0.8 + 0.2 * cos(150.0 * d);
    col = mix(col, vec3(1.0), 1.0 - smoothstep(0.0, 0.01, abs(d)));

    fragColor = vec4(col, 1.0);
}