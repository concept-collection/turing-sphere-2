/**
 * WGSL Legendre-transform kernels, modeled on leg_m_kernel / ileg_m_kernel
 * in SHT/cuda_legendre.gen.cu (non-Ishioka fp32 path: SHTNS disables the
 * Ishioka recurrence for fp32 because it loses too much accuracy).
 *
 * Synthesis:  F_m(theta_i) = sum_{l=m..lmax} Q_lm * ytilde_l^m(theta_i)
 *   - one thread per latitude, one workgroup row per m (workgroup_id.y).
 * Analysis:   Q_lm = sum_i w_i * G_m(theta_i) * ytilde_l^m(theta_i)
 *   - one workgroup per m; threads own latitudes (strided); per-l pair
 *     workgroup tree reduction (portable stand-in for the CUDA warp
 *     shuffles).
 *
 * The associated Legendre functions are generated on the fly by the
 * standard 3-term recurrence over l (coefficients a,b precomputed on the
 * host in f64), with the SHTNS fp32 rescaling scheme for sin(theta)^m
 * underflow (see common.ts).
 */
import { RESCALE_WGSL } from './common.ts';

export interface LegParams {
  lmax: number;
  mmax: number;
  nlat: number;
  wgSynth: number; // workgroup size for synthesis (threads over latitude)
  wgAnalys: number; // workgroup size for analysis (power of two)
}

const BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<storage, read> ab: array<vec2f>;     // (a_l^m, b_l^m) per lm
@group(0) @binding(1) var<storage, read> amm: array<f32>;      // seed per m
@group(0) @binding(2) var<storage, read> ctstw: array<f32>;    // [ct | st | w], each NLAT
`;

export function legSynthWGSL(p: LegParams): string {
  return /* wgsl */ `
${RESCALE_WGSL}
const LMAX: u32 = ${p.lmax}u;
const NLAT: u32 = ${p.nlat}u;
${BINDINGS}
@group(0) @binding(3) var<storage, read> qlm: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> fm: array<vec2f>;  // [(m)*NLAT + ilat]

@compute @workgroup_size(${p.wgSynth})
fn leg_synth(@builtin(global_invocation_id) gid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let ilat = gid.x;
  let m = wid.y;
  if (ilat >= NLAT) { return; }

  let ct = ctstw[ilat];
  let st = ctstw[NLAT + ilat];
  let base = m * (LMAX + 1u) - (m * (m - 1u)) / 2u;   // lm index of (l=m, m)

  var seed = sinpow_rescaled(st, m);
  var y0 = seed.y0 * amm[m];
  var ny = seed.ny;
  var y1: f32 = 0.0;
  if (m < LMAX) {
    y1 = ab[base + 1u].x * ct * y0;
  }

  var acc = vec2f(0.0);
  var l = m;
  loop {
    if (ny == 0) {
      acc += y0 * qlm[base + (l - m)];
      if (l + 1u <= LMAX) {
        acc += y1 * qlm[base + (l + 1u - m)];
      }
    } else if (abs(y0) > RESCALE_THR) {
      ny += 1;
      y0 *= INV_SCALE;
      y1 *= INV_SCALE;
    }
    if (l + 2u > LMAX) { break; }
    let c0 = ab[base + (l + 2u - m)];
    var c1 = vec2f(0.0);
    if (l + 3u <= LMAX) {
      c1 = ab[base + (l + 3u - m)];
    }
    // Route the new y0 through an explicit temporary, exactly as leg_analys
    // does, rather than assigning y0 and reading it back on the next line:
    // NVIDIA's WGSL compiler schedules the y1 update against the OLD y0 there,
    // which lags the recurrence by one step and silently corrupts synthesis on
    // hardware (it is correct under SwiftShader, so CI never saw it).
    // Mirrors concept-collection/shtns-webgpu#1; this file is a vendored copy
    // of that library's src/wgsl/leg.ts, so the fix has to be made in both.
    let t0 = c0.x * ct * y1 + c0.y * y0;
    y1 = c1.x * ct * t0 + c1.y * y1;
    y0 = t0;
    l += 2u;
  }
  fm[m * NLAT + ilat] = acc;
}
`;
}

export function legAnalysWGSL(p: LegParams): string {
  const K = Math.ceil(p.nlat / p.wgAnalys); // latitudes per thread
  return /* wgsl */ `
${RESCALE_WGSL}
const LMAX: u32 = ${p.lmax}u;
const NLAT: u32 = ${p.nlat}u;
const WG: u32 = ${p.wgAnalys}u;
const K: u32 = ${K}u;
${BINDINGS}
@group(0) @binding(3) var<storage, read> fm: array<vec2f>;        // [(m)*NLAT + ilat]
@group(0) @binding(4) var<storage, read_write> qout: array<vec2f>;

var<workgroup> red: array<vec4f, ${p.wgAnalys}>;

@compute @workgroup_size(${p.wgAnalys})
fn leg_analys(@builtin(local_invocation_id) lid3: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let m = wid.x;
  let base = m * (LMAX + 1u) - (m * (m - 1u)) / 2u;

  // per-thread recurrence state for K latitudes
  var y0v: array<f32, ${K}>;
  var y1v: array<f32, ${K}>;
  var nyv: array<i32, ${K}>;
  var ctv: array<f32, ${K}>;
  var wfv: array<vec2f, ${K}>;

  for (var k = 0u; k < K; k++) {
    let lat = lid + k * WG;
    var ct: f32 = 0.0;
    var st: f32 = 0.0;
    var wf = vec2f(0.0);
    if (lat < NLAT) {
      ct = ctstw[lat];
      st = ctstw[NLAT + lat];
      wf = fm[m * NLAT + lat] * ctstw[2u * NLAT + lat];  // Gauss weight (incl. 2*pi/nphi)
    }
    ctv[k] = ct;
    let seed = sinpow_rescaled(st, m);
    y0v[k] = seed.y0 * amm[m];
    nyv[k] = seed.ny;
    y1v[k] = 0.0;
    if (m < LMAX) {
      y1v[k] = ab[base + 1u].x * ct * y0v[k];
    }
    wfv[k] = wf;
  }

  var l = m;
  loop {
    var c0 = vec2f(0.0);
    var c1 = vec2f(0.0);
    for (var k = 0u; k < K; k++) {
      if (nyv[k] == 0) {
        c0 += wfv[k] * y0v[k];
        c1 += wfv[k] * y1v[k];
      } else if (abs(y0v[k]) > RESCALE_THR) {
        nyv[k] += 1;
        y0v[k] *= INV_SCALE;
        y1v[k] *= INV_SCALE;
      }
    }
    // workgroup tree reduction of (c0, c1)
    red[lid] = vec4f(c0, c1);
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
      if (lid < s) { red[lid] += red[lid + s]; }
      workgroupBarrier();
      s = s >> 1u;
    }
    if (lid == 0u) {
      qout[base + (l - m)] = red[0].xy;
      if (l + 1u <= LMAX) {
        qout[base + (l + 1u - m)] = red[0].zw;
      }
    }
    if (l + 2u > LMAX) { break; }
    let a0 = ab[base + (l + 2u - m)];
    var a1 = vec2f(0.0);
    if (l + 3u <= LMAX) {
      a1 = ab[base + (l + 3u - m)];
    }
    for (var k = 0u; k < K; k++) {
      let t0 = a0.x * ctv[k] * y1v[k] + a0.y * y0v[k];
      y0v[k] = t0;
      y1v[k] = a1.x * ctv[k] * t0 + a1.y * y1v[k];
    }
    l += 2u;
  }
}
`;
}
