/**
 * Custom WebGL node programs for distinct symbol-type shapes.
 *
 * Each shape subclasses NodeCircleProgram and overrides getDefinition()
 * to swap in a custom fragment shader with a shape-specific SDF.
 * The vertex shader, attributes, uniforms, and processVisibleItem
 * are all inherited — only the fragment shader changes.
 */
import { NodeCircleProgram, NodeProgram } from 'sigma/rendering';
import { floatColor } from 'sigma/utils';

export { NodeCircleProgram };

// ─── Shared vertex shader (identical to NodeCircleProgram's) ──────────
const VERTEX_SHADER = /* glsl */ `
attribute vec4 a_id;
attribute vec4 a_color;
attribute vec2 a_position;
attribute float a_size;
attribute float a_angle;

uniform mat3 u_matrix;
uniform float u_sizeRatio;
uniform float u_correctionRatio;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;

const float bias = 255.0 / 254.0;

void main() {
  float size = a_size * u_correctionRatio / u_sizeRatio * 4.0;
  vec2 diffVector = size * vec2(cos(a_angle), sin(a_angle));
  vec2 position = a_position + diffVector;
  gl_Position = vec4(
    (u_matrix * vec3(position, 1)).xy,
    0,
    1
  );

  v_diffVector = diffVector;
  v_radius = size / 2.0;

  #ifdef PICKING_MODE
  v_color = a_id;
  #else
  v_color = a_color;
  #endif

  v_color.a *= bias;
}
`;

// ─── Fragment shader: Diamond (rotated square SDF) ────────────────────
const FRAG_DIAMOND = /* glsl */ `
precision highp float;
varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;
uniform float u_correctionRatio;
const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float border = u_correctionRatio * 2.0;
  // Diamond SDF: |x| + |y| <= radius
  float dist = (abs(v_diffVector.x) + abs(v_diffVector.y)) - v_radius + border;

  #ifdef PICKING_MODE
  if (dist > border) gl_FragColor = transparent;
  else gl_FragColor = v_color;
  #else
  float t = 0.0;
  if (dist > border) t = 1.0;
  else if (dist > 0.0) t = dist / border;
  gl_FragColor = mix(v_color, transparent, t);
  #endif
}
`;

// ─── Fragment shader: Hexagon SDF ─────────────────────────────────────
const FRAG_HEXAGON = /* glsl */ `
precision highp float;
varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;
uniform float u_correctionRatio;
const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float border = u_correctionRatio * 2.0;
  vec2 p = abs(v_diffVector);
  // Hexagon SDF: max(p.x + p.y * 0.577, p.y * 1.155) - radius
  float dist = max(p.x + p.y * 0.57735, p.y * 1.1547) - v_radius + border;

  #ifdef PICKING_MODE
  if (dist > border) gl_FragColor = transparent;
  else gl_FragColor = v_color;
  #else
  float t = 0.0;
  if (dist > border) t = 1.0;
  else if (dist > 0.0) t = dist / border;
  gl_FragColor = mix(v_color, transparent, t);
  #endif
}
`;

// ─── Fragment shader: Rounded square SDF ──────────────────────────────
const FRAG_SQUARE = /* glsl */ `
precision highp float;
varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;
uniform float u_correctionRatio;
const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float border = u_correctionRatio * 2.0;
  float side = v_radius * 0.8;
  float corner = v_radius * 0.2;
  vec2 p = abs(v_diffVector);
  // Rounded rectangle SDF
  vec2 q = p - vec2(side - corner);
  float dist = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - corner - border;
  dist += border;

  #ifdef PICKING_MODE
  if (dist > border) gl_FragColor = transparent;
  else gl_FragColor = v_color;
  #else
  float t = 0.0;
  if (dist > border) t = 1.0;
  else if (dist > 0.0) t = dist / border;
  gl_FragColor = mix(v_color, transparent, t);
  #endif
}
`;

// ─── Fragment shader: Triangle (pointing up) ──────────────────────────
const FRAG_TRIANGLE = /* glsl */ `
precision highp float;
varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;
uniform float u_correctionRatio;
const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float border = u_correctionRatio * 2.0;
  float r = v_radius;
  vec2 p = v_diffVector;
  // Equilateral triangle SDF (pointing up), centered
  p.y = -p.y + r * 0.2;
  p.x = abs(p.x);
  float dist = max(
    p.x * 0.866 + p.y * 0.5,
    -p.y
  ) - r * 0.5 + border;

  #ifdef PICKING_MODE
  if (dist > border) gl_FragColor = transparent;
  else gl_FragColor = v_color;
  #else
  float t = 0.0;
  if (dist > border) t = 1.0;
  else if (dist > 0.0) t = dist / border;
  gl_FragColor = mix(v_color, transparent, t);
  #endif
}
`;

// ─── Factory: create a NodeProgram class with a custom frag shader ────
const { FLOAT, UNSIGNED_BYTE } = WebGLRenderingContext;
const UNIFORMS = ['u_sizeRatio', 'u_correctionRatio', 'u_matrix'];
const ANGLE_1 = 0;
const ANGLE_2 = (2 * Math.PI) / 3;
const ANGLE_3 = (4 * Math.PI) / 3;

function createShapeNodeProgram(fragmentShaderSource: string) {
  return class ShapeNodeProgram extends NodeProgram {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    getDefinition(): any {
      return {
        VERTICES: 3,
        VERTEX_SHADER_SOURCE: VERTEX_SHADER,
        FRAGMENT_SHADER_SOURCE: fragmentShaderSource,
        METHOD: WebGLRenderingContext.TRIANGLES,
        UNIFORMS,
        ATTRIBUTES: [
          { name: 'a_position', size: 2, type: FLOAT },
          { name: 'a_size', size: 1, type: FLOAT },
          { name: 'a_color', size: 4, type: UNSIGNED_BYTE, normalized: true },
          { name: 'a_id', size: 4, type: UNSIGNED_BYTE, normalized: true },
        ],
        CONSTANT_ATTRIBUTES: [{ name: 'a_angle', size: 1, type: FLOAT }],
        CONSTANT_DATA: [[ANGLE_1], [ANGLE_2], [ANGLE_3]],
      };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    processVisibleItem(nodeIndex: number, startIndex: number, data: any): void {
      const array = this.array;
      array[startIndex++] = data.x;
      array[startIndex++] = data.y;
      array[startIndex++] = data.size;
      array[startIndex++] = floatColor(data.color);
      array[startIndex] = nodeIndex;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setUniforms(params: any, { gl, uniformLocations }: any): void {
      gl.uniform1f(uniformLocations.u_correctionRatio, params.correctionRatio);
      gl.uniform1f(uniformLocations.u_sizeRatio, params.sizeRatio);
      gl.uniformMatrix3fv(uniformLocations.u_matrix, false, params.matrix);
    }
  };
}

// ─── Exported programs ────────────────────────────────────────────────
export const NodeDiamondProgram = createShapeNodeProgram(FRAG_DIAMOND);
export const NodeHexagonProgram = createShapeNodeProgram(FRAG_HEXAGON);
export const NodeSquareProgram = createShapeNodeProgram(FRAG_SQUARE);
export const NodeTriangleProgram = createShapeNodeProgram(FRAG_TRIANGLE);

// Map symbol types to node `type` strings (used by nodeReducer)
export const SYMBOL_TYPE_MAP: Record<string, string> = {
  module: 'hexagon',
  class: 'diamond',
  function: 'circle',
  method: 'circle',
  variable: 'square',
  parameter: 'triangle',
};

// Map node `type` strings to WebGL programs (used by Sigma nodeProgramClasses)
export const NODE_PROGRAM_CLASSES: Record<string, typeof NodeProgram> = {
  circle: NodeCircleProgram as unknown as typeof NodeProgram,
  diamond: NodeDiamondProgram as unknown as typeof NodeProgram,
  hexagon: NodeHexagonProgram as unknown as typeof NodeProgram,
  square: NodeSquareProgram as unknown as typeof NodeProgram,
  triangle: NodeTriangleProgram as unknown as typeof NodeProgram,
};
