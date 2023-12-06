#version 330 core

out vec4 outColor;

in vec3 fragNormal;
in vec3 fragPosition;

uniform vec4 light_pos;
uniform vec3 eye_pos;
uniform float ID;
uniform vec3 material_color;

uniform vec3 specular_color;
uniform int shininess;
uniform float ambient_intensity;
uniform float K_s;
uniform bool sil;

vec3 computeToon2(vec3 L, vec3 N){
      float intensity = clamp(dot(L, N), 0 ,1);
      int n = 4;

      float step = sqrt(intensity) * n;
      intensity = (floor(step) + smoothstep(0.48, 0.52, fract(step))) / n;
      intensity = intensity * intensity;

      return material_color * intensity;
}

void main(){
      vec3 N = normalize(fragNormal);
      vec3 L; //Light_dir
      if (light_pos.w==0.0)   L = normalize(light_pos.xyz);                   // directional light
      else                    L = normalize(light_pos.xyz-fragPosition);      // point light

      vec3 res = computeToon2(N, L);
      if(sil == true && dot(N, normalize(eye_pos-fragPosition)) < 0.2)
      {
            res = vec3(0,0,0);
      }

      outColor = vec4(res, 1.0);
}