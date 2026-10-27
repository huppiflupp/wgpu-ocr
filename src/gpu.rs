//! Both ocrs models on wgpu compute shaders, driven by the ONNX graph read in Rust.
//!
//! - `Graph` runs the convolutional graphs node by node: Conv (dense and depthwise), ConvTranspose,
//!   MaxPool, Slice, Concat, Pad (zero only), Relu/Sigmoid (fused into the convolution before them).
//! - `Detector` is the whole detection model as a `Graph`.
//! - `Recognizer` is the recognition model's convolutional trunk as a `Graph`, then the two
//!   bidirectional GRU layers (one dispatch per layer for all lines, each line with its own
//!   length) and the output layer.

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::onnx::{Node, Onnx};

type Buf = Arc<wgpu::Buffer>;

const EPILOGUE: &str = r#"
fn erf_approx(v: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7.
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-a * a));
}
fn activate(v: f32, act: u32) -> f32 {
    if (act == 1u) { return max(v, 0.0); }
    if (act == 2u) { return 1.0 / (1.0 + exp(-v)); }
    if (act == 3u) { return 0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752)); }
    if (act == 4u) { return v / (1.0 + exp(-v)); }
    return v;
}
"#;

/// Dense convolution as an implicit GEMM, 64 pixels x 64 channels per workgroup, 4x4 per thread.
/// FAST: cin is a multiple of 16 and the weights are ordered [ky][kx][ci][co], so a 16-deep
/// slice of K has one tap (ky, kx) and 16 consecutive input channels: no division per element.
/// Otherwise the weights are [ci][ky][kx][co] and each element decodes its k.
const CONV: &str = r#"
struct P { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, kh: u32, kw: u32, pad: u32, stride: u32, act: u32, padx: u32, stridex: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> res: array<f32>;  // added to the output when p.x0 == 1
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
EPILOGUE_FN

fn put(m: u32, c0: u32, v: vec4<f32>) {
    if (m >= p.n * p.oh * p.ow) { return; }
    let ox = m % p.ow;
    let t = m / p.ow;
    let oy = t % p.oh;
    let nb = t / p.oh;
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < p.cout) {
            let o = ((nb * p.cout + c) * p.oh + oy) * p.ow + ox;
            var r = activate(v[j] + b[c], p.act);
            if (p.x0 == 1u) { r += res[o]; }
            y[o] = r;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 64u;
    let c0 = wid.y * 64u;
    let mtot = p.n * p.oh * p.ow;
    let ktot = p.cin * p.kh * p.kw;
    // The A elements this thread loads are always the same pixel (idx % 64 == tid % 64).
    let mm = tid % 64u;
    let kq = tid / 64u;
    let m = m0 + mm;
    let valid = m < mtot;
    let ox = m % p.ow;
    let tt = m / p.ow;
    let oy = tt % p.oh;
    let nb = tt / p.oh;
    let iy0 = i32(oy * p.stride) - i32(p.pad);
    let ix0 = i32(ox * p.stridex) - i32(p.padx);
    let plane = p.h * p.w;
    let xbase = nb * p.cin * plane;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < ktot; k0 += 16u) {
FAST_OR_GENERIC
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let ab = kk * 64u + lid.x * 4u;
            let a = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
            let bb = kk * 64u + lid.y * 4u;
            let bv = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
            acc0 += a.x * bv;
            acc1 += a.y * bv;
            acc2 += a.z * bv;
            acc3 += a.w * bv;
        }
        workgroupBarrier();
    }
    let mb = m0 + lid.x * 4u;
    let cb = c0 + lid.y * 4u;
    put(mb, cb, acc0);
    put(mb + 1u, cb, acc1);
    put(mb + 2u, cb, acc2);
    put(mb + 3u, cb, acc3);
}
"#;

/// The fast path with a 128-pixel x 64-channel tile, 8 x 4 outputs per thread, and the tiles
/// read from workgroup memory as vec4: 3 loads per 32 FMAs instead of 8 per 16.
const CONV128: &str = r#"
struct P { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, kh: u32, kw: u32, pad: u32, stride: u32, act: u32, padx: u32, stridex: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> res: array<f32>;  // added to the output when p.x0 == 1
var<workgroup> As: array<vec4<f32>, 512>;   // [16 k][128 m / 4]
var<workgroup> Bs: array<vec4<f32>, 256>;   // [16 k][64 c / 4]
EPILOGUE_FN

fn put(m: u32, c0: u32, v: vec4<f32>) {
    if (m >= p.n * p.oh * p.ow) { return; }
    let ox = m % p.ow;
    let t = m / p.ow;
    let oy = t % p.oh;
    let nb = t / p.oh;
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < p.cout) {
            let o = ((nb * p.cout + c) * p.oh + oy) * p.ow + ox;
            var r = activate(v[j] + b[c], p.act);
            if (p.x0 == 1u) { r += res[o]; }
            y[o] = r;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 128u;
    let c0 = wid.y * 64u;
    let mtot = p.n * p.oh * p.ow;
    let ktot = p.cin * p.kh * p.kw;
    // A: this thread always loads pixel tid % 128, at k rows tid / 128 + 2e.
    let mm = tid % 128u;
    let ka = tid / 128u;
    let m = m0 + mm;
    let valid = m < mtot;
    let ox = m % p.ow;
    let tt = m / p.ow;
    let oy = tt % p.oh;
    let nb = tt / p.oh;
    let iy0 = i32(oy * p.stride) - i32(p.pad);
    let ix0 = i32(ox * p.stridex) - i32(p.padx);
    let plane = p.h * p.w;
    let xbase = nb * p.cin * plane;
    // B: channel tid % 64, k rows tid / 64 + 4e.
    let cc = tid % 64u;
    let kb = tid / 64u;
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < ktot; k0 += 16u) {
        let tap = k0 / p.cin;
        let ci0 = k0 % p.cin;
        let iy = iy0 + i32(tap / p.kw);
        let ix = ix0 + i32(tap % p.kw);
        let inside = valid && iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w);
        let pix = u32(iy) * p.w + u32(ix);
        for (var e = 0u; e < 8u; e++) {
            let kk = ka + e * 2u;
            var v = 0.0;
            if (inside) { v = x[xbase + (ci0 + kk) * plane + pix]; }
            As[kk * 32u + mm / 4u][mm % 4u] = v;
        }
        for (var e = 0u; e < 4u; e++) {
            let kk = kb + e * 4u;
            var wv = 0.0;
            if (c0 + cc < p.cout) { wv = wt[(k0 + kk) * p.cout + c0 + cc]; }
            Bs[kk * 16u + cc / 4u][cc % 4u] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let lo = As[kk * 32u + lid.x * 2u];
            let hi = As[kk * 32u + lid.x * 2u + 1u];
            let bv = Bs[kk * 16u + lid.y];
            a0 += lo.x * bv; a1 += lo.y * bv; a2 += lo.z * bv; a3 += lo.w * bv;
            a4 += hi.x * bv; a5 += hi.y * bv; a6 += hi.z * bv; a7 += hi.w * bv;
        }
        workgroupBarrier();
    }
    let mb = m0 + lid.x * 8u;
    let cb = c0 + lid.y * 4u;
    put(mb, cb, a0); put(mb + 1u, cb, a1); put(mb + 2u, cb, a2); put(mb + 3u, cb, a3);
    put(mb + 4u, cb, a4); put(mb + 5u, cb, a5); put(mb + 6u, cb, a6); put(mb + 7u, cb, a7);
}
"#;

const CONV64V: &str = r#"
struct P { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, kh: u32, kw: u32, pad: u32, stride: u32, act: u32, padx: u32, stridex: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> res: array<f32>;  // added to the output when p.x0 == 1
var<workgroup> As: array<vec4<f32>, 256>;   // [16 k][64 m / 4]
var<workgroup> Bs: array<vec4<f32>, 256>;   // [16 k][64 c / 4]
EPILOGUE_FN

fn put(m: u32, c0: u32, v: vec4<f32>) {
    if (m >= p.n * p.oh * p.ow) { return; }
    let ox = m % p.ow;
    let t = m / p.ow;
    let oy = t % p.oh;
    let nb = t / p.oh;
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < p.cout) {
            let o = ((nb * p.cout + c) * p.oh + oy) * p.ow + ox;
            var r = activate(v[j] + b[c], p.act);
            if (p.x0 == 1u) { r += res[o]; }
            y[o] = r;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 64u;
    let c0 = wid.y * 64u;
    let mtot = p.n * p.oh * p.ow;
    let ktot = p.cin * p.kh * p.kw;
    // A: this thread always loads pixel tid % 64, at k rows tid / 64 + 4e.
    let mm = tid % 64u;
    let ka = tid / 64u;
    let m = m0 + mm;
    let valid = m < mtot;
    let ox = m % p.ow;
    let tt = m / p.ow;
    let oy = tt % p.oh;
    let nb = tt / p.oh;
    let iy0 = i32(oy * p.stride) - i32(p.pad);
    let ix0 = i32(ox * p.stridex) - i32(p.padx);
    let plane = p.h * p.w;
    let xbase = nb * p.cin * plane;
    // B: channel tid % 64, k rows tid / 64 + 4e.
    let cc = tid % 64u;
    let kb = tid / 64u;
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < ktot; k0 += 16u) {
        let tap = k0 / p.cin;
        let ci0 = k0 % p.cin;
        let iy = iy0 + i32(tap / p.kw);
        let ix = ix0 + i32(tap % p.kw);
        let inside = valid && iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w);
        let pix = u32(iy) * p.w + u32(ix);
        for (var e = 0u; e < 4u; e++) {
            let kk = ka + e * 4u;
            var v = 0.0;
            if (inside) { v = x[xbase + (ci0 + kk) * plane + pix]; }
            As[kk * 16u + mm / 4u][mm % 4u] = v;
        }
        for (var e = 0u; e < 4u; e++) {
            let kk = kb + e * 4u;
            var wv = 0.0;
            if (c0 + cc < p.cout) { wv = wt[(k0 + kk) * p.cout + c0 + cc]; }
            Bs[kk * 16u + cc / 4u][cc % 4u] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let lo = As[kk * 16u + lid.x];
            let bv = Bs[kk * 16u + lid.y];
            a0 += lo.x * bv; a1 += lo.y * bv; a2 += lo.z * bv; a3 += lo.w * bv;
        }
        workgroupBarrier();
    }
    let mb = m0 + lid.x * 4u;
    let cb = c0 + lid.y * 4u;
    put(mb, cb, a0); put(mb + 1u, cb, a1); put(mb + 2u, cb, a2); put(mb + 3u, cb, a3);
}
"#;

const CONV_FAST_LOAD: &str = r#"
        let tap = k0 / p.cin;
        let ci0 = k0 % p.cin;
        let iy = iy0 + i32(tap / p.kw);
        let ix = ix0 + i32(tap % p.kw);
        let inside = valid && iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w);
        let pix = u32(iy) * p.w + u32(ix);
        for (var e = 0u; e < 4u; e++) {
            let kk = kq + e * 4u;
            var v = 0.0;
            if (inside) { v = x[xbase + (ci0 + kk) * plane + pix]; }
            As[kk * 64u + mm] = v;
            let c = c0 + mm;
            var wv = 0.0;
            if (c < p.cout) { wv = wt[(k0 + kk) * p.cout + c]; }
            Bs[kk * 64u + mm] = wv;
        }
"#;

const CONV_GENERIC_LOAD: &str = r#"
        let khw = p.kh * p.kw;
        for (var e = 0u; e < 4u; e++) {
            let kk = kq + e * 4u;
            let k = k0 + kk;
            var v = 0.0;
            if (valid && k < ktot) {
                let ci = k / khw;
                let r = k % khw;
                let iy = iy0 + i32(r / p.kw);
                let ix = ix0 + i32(r % p.kw);
                if (iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w)) {
                    v = x[xbase + ci * plane + u32(iy) * p.w + u32(ix)];
                }
            }
            As[kk * 64u + mm] = v;
            let c = c0 + mm;
            var wv = 0.0;
            if (k < ktot && c < p.cout) { wv = wt[k * p.cout + c]; }
            Bs[kk * 64u + mm] = wv;
        }
"#;

/// One thread per output element; `i` from a 2D grid of 256-wide workgroups.
const INDEX: &str = "let i = (wid.y * nw.x + wid.x) * 256u + lid.x;";
const ENTRY: &str = "@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {";

const DEPTHWISE: &str = r#"
struct P { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, kh: u32, kw: u32, pad: u32, stride: u32, act: u32, padx: u32, stridex: u32, x0: u32, x1: u32, x2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;   // [c][kh][kw]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
EPILOGUE_FN
ENTRY
    INDEX
    if (i >= p.n * p.c * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let oy = (i / p.ow) % p.oh;
    let nc = i / (p.ow * p.oh);
    let c = nc % p.c;
    var acc = b[c];
    for (var ky = 0u; ky < p.kh; ky++) {
        let iy = i32(oy * p.stride + ky) - i32(p.pad);
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0u; kx < p.kw; kx++) {
            let ix = i32(ox * p.stridex + kx) - i32(p.padx);
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            acc += x[(nc * p.h + u32(iy)) * p.w + u32(ix)] * wt[(c * p.kh + ky) * p.kw + kx];
        }
    }
    y[i] = activate(acc, p.act);
}
"#;

/// A convolution with one input channel, direct (no GEMM tile to waste), with its activation
/// and, if one follows, its MaxPool fused: the full-size output is never written.
const CONV_DIRECT: &str = r#"
struct P { n: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, kh: u32, kw: u32, pad: u32, act: u32, ph: u32, pw: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;   // [cout][kh][kw]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
EPILOGUE_FN
ENTRY
    INDEX
    if (i >= p.n * p.cout * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let oy = (i / p.ow) % p.oh;
    let co = (i / (p.ow * p.oh)) % p.cout;
    let nb = i / (p.ow * p.oh * p.cout);
    var best = -3.4e38;
    for (var py = 0u; py < p.ph; py++) {
        for (var px = 0u; px < p.pw; px++) {
            let cy = oy * p.ph + py;
            let cx = ox * p.pw + px;
            var acc = b[co];
            for (var ky = 0u; ky < p.kh; ky++) {
                let iy = i32(cy + ky) - i32(p.pad);
                if (iy < 0 || iy >= i32(p.h)) { continue; }
                for (var kx = 0u; kx < p.kw; kx++) {
                    let ix = i32(cx + kx) - i32(p.pad);
                    if (ix < 0 || ix >= i32(p.w)) { continue; }
                    acc += x[(nb * p.h + u32(iy)) * p.w + u32(ix)] * wt[(co * p.kh + ky) * p.kw + kx];
                }
            }
            best = max(best, activate(acc, p.act));
        }
    }
    y[i] = best;
}
"#;

const CONV_TRANSPOSE: &str = r#"
struct P { n: u32, cin: u32, h: u32, w: u32, cout: u32, oh: u32, ow: u32, kh: u32, kw: u32, stride: u32, act: u32, x0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;   // [cin][cout][kh][kw]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
EPILOGUE_FN
ENTRY
    INDEX
    if (i >= p.n * p.cout * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let oy = (i / p.ow) % p.oh;
    let co = (i / (p.ow * p.oh)) % p.cout;
    let nb = i / (p.ow * p.oh * p.cout);
    var acc = b[co];
    // Only the taps of this pixel's parity: ky = oy mod stride, + stride, ...
    let plane = p.h * p.w;
    let wstep = p.cout * p.kh * p.kw;
    for (var ky = oy % p.stride; ky < p.kh && ky <= oy; ky += p.stride) {
        let iy = (oy - ky) / p.stride;
        if (iy >= p.h) { continue; }
        for (var kx = ox % p.stride; kx < p.kw && kx <= ox; kx += p.stride) {
            let ix = (ox - kx) / p.stride;
            if (ix >= p.w) { continue; }
            var xi = nb * p.cin * plane + iy * p.w + ix;
            var wi = (co * p.kh + ky) * p.kw + kx;
            for (var ci = 0u; ci < p.cin; ci++) {
                acc += x[xi] * wt[wi];
                xi += plane;
                wi += wstep;
            }
        }
    }
    y[i] = activate(acc, p.act);
}
"#;

const MAXPOOL: &str = r#"
struct P { nc: u32, h: u32, w: u32, oh: u32, ow: u32, kh: u32, kw: u32, x0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.nc * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let t = i / p.ow;
    let oy = t % p.oh;
    let c = t / p.oh;
    var m = -3.4e38;
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            m = max(m, x[(c * p.h + oy * p.kh + ky) * p.w + ox * p.kw + kx]);
        }
    }
    y[i] = m;
}
"#;

/// out[n][c][y][x] = in[n + s.x][c + s.y][y + s.z][x + s.w]
const SLICE: &str = r#"
struct P { i: vec4<u32>, o: vec4<u32>, s: vec4<u32> }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.o.x * p.o.y * p.o.z * p.o.w) { return; }
    let xx = i % p.o.w;
    let yy = (i / p.o.w) % p.o.z;
    let cc = (i / (p.o.w * p.o.z)) % p.o.y;
    let nn = i / (p.o.w * p.o.z * p.o.y);
    y[i] = x[(((nn + p.s.x) * p.i.y + cc + p.s.y) * p.i.z + yy + p.s.z) * p.i.w + xx + p.s.w];
}
"#;

/// Copy x [n][c][hw] into channels [off, off + c) of y [n][ctot][hw].
const CONCAT: &str = r#"
struct P { n: u32, c: u32, hw: u32, ctot: u32, off: u32, x0: u32, x1: u32, x2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.n * p.c * p.hw) { return; }
    let s = i % p.hw;
    let c = (i / p.hw) % p.c;
    let nb = i / (p.hw * p.c);
    y[(nb * p.ctot + p.off + c) * p.hw + s] = x[i];
}
"#;

const ACTIVATE: &str = r#"
struct P { count: u32, act: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
EPILOGUE_FN
ENTRY
    INDEX
    if (i >= p.count) { return; }
    y[i] = activate(x[i], p.act);
}
"#;

/// RGB bytes of the page (packed 4 to a word) to ocrs' grey values, padded with black to at
/// least the model size, and resized to it bilinearly as rten's `resize_image` does
/// (half-pixel coordinates, clamped, x then y).
const GREY_RESIZE: &str = r#"
struct P { w: u32, h: u32, pw: u32, ph: u32, ow: u32, oh: u32, sx: f32, sy: f32 }
@group(0) @binding(0) var<storage, read> rgb: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
fn byte(i: u32) -> f32 { return f32((rgb[i / 4u] >> ((i % 4u) * 8u)) & 255u); }
fn grey(x: u32, yy: u32) -> f32 {
    if (x >= p.w || yy >= p.h) { return -0.5; }
    let i = (yy * p.w + x) * 3u;
    return -0.5 + byte(i) * (0.299 / 255.0) + byte(i + 1u) * (0.587 / 255.0) + byte(i + 2u) * (0.114 / 255.0);
}
ENTRY
    INDEX
    if (i >= p.ow * p.oh) { return; }
    let ox = i % p.ow;
    let oy = i / p.ow;
    let iy = clamp(p.sy * (f32(oy) + 0.5) - 0.5, 0.0, f32(p.ph) - 1.0);
    let ix = clamp(p.sx * (f32(ox) + 0.5) - 0.5, 0.0, f32(p.pw) - 1.0);
    let y1 = u32(iy);
    let x1 = u32(ix);
    let y2 = min(y1 + 1u, p.ph - 1u);
    let x2 = min(x1 + 1u, p.pw - 1u);
    let wy = iy - f32(y1);
    let wx = ix - f32(x1);
    let top = mix(grey(x1, y1), grey(x2, y1), wx);
    let bottom = mix(grey(x1, y2), grey(x2, y2), wx);
    y[i] = mix(top, bottom, wy);
}
"#;

/// The model's mask (its first `mh` x `mw` part, without the padding) resized to the page as
/// rten does, thresholded, and packed 32 pixels to a word per row.
const MASK_UP: &str = r#"
struct P { stride: u32, mw: u32, mh: u32, w: u32, h: u32, words: u32, sx: f32, sy: f32, thr: f32, x0: u32, x1: u32, x2: u32 }
@group(0) @binding(0) var<storage, read> m: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.words * p.h) { return; }
    let wx = i % p.words;
    let oy = i / p.words;
    let iy = clamp(p.sy * (f32(oy) + 0.5) - 0.5, 0.0, f32(p.mh) - 1.0);
    let y1 = u32(iy);
    let y2 = min(y1 + 1u, p.mh - 1u);
    let fy = iy - f32(y1);
    var bits = 0u;
    for (var b = 0u; b < 32u; b++) {
        let ox = wx * 32u + b;
        if (ox >= p.w) { break; }
        let ix = clamp(p.sx * (f32(ox) + 0.5) - 0.5, 0.0, f32(p.mw) - 1.0);
        let x1 = u32(ix);
        let x2 = min(x1 + 1u, p.mw - 1u);
        let fx = ix - f32(x1);
        let top = mix(m[y1 * p.stride + x1], m[y1 * p.stride + x2], fx);
        let bottom = mix(m[y2 * p.stride + x1], m[y2 * p.stride + x2], fx);
        if (mix(top, bottom, fy) > p.thr) { bits |= 1u << b; }
    }
    y[i] = bits;
}
"#;

/// AvgPool (kh, 1) to one output row, written into the GRU input [t][line][c] at this batch's
/// line offset.
const AVGPOOL_T: &str = r#"
struct P { n: u32, c: u32, h: u32, w: u32, kh: u32, ntot: u32, noff: u32, x0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.n * p.c * p.w) { return; }
    let c = i % p.c;
    let t = (i / p.c) % p.w;
    let nb = i / (p.c * p.w);
    var s = 0.0;
    for (var r = 0u; r < p.kh; r++) {
        s += x[((nb * p.c + c) * p.h + r) * p.w + t];
    }
    y[(t * p.ntot + p.noff + nb) * p.c + c] = s / f32(p.kh);
}
"#;

/// y[m, k] = x[m, :] . wt[:, k] + b[k] with wt stored [i][k].
const GEMM: &str = r#"
struct Dims { m: u32, k: u32, i: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

fn put(m: u32, c0: u32, v: vec4<f32>) {
    if (m >= d.m) { return; }
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < d.k) { y[m * d.k + c] = v[j] + b[c]; }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 64u;
    let c0 = wid.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < d.i; k0 += 16u) {
        for (var e = 0u; e < 4u; e++) {
            let idx = tid + e * 256u;
            let am = idx / 16u;
            let ak = idx % 16u;
            var v = 0.0;
            if (m0 + am < d.m && k0 + ak < d.i) { v = x[(m0 + am) * d.i + k0 + ak]; }
            As[ak * 64u + am] = v;
            let bk = idx / 64u;
            let bc = idx % 64u;
            var wv = 0.0;
            if (k0 + bk < d.i && c0 + bc < d.k) { wv = wt[(k0 + bk) * d.k + c0 + bc]; }
            Bs[bk * 64u + bc] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let ab = kk * 64u + lid.x * 4u;
            let a = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
            let bb = kk * 64u + lid.y * 4u;
            let bv = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
            acc0 += a.x * bv;
            acc1 += a.y * bv;
            acc2 += a.z * bv;
            acc3 += a.w * bv;
        }
        workgroupBarrier();
    }
    let mb = m0 + lid.x * 4u;
    let cb = c0 + lid.y * 4u;
    put(mb, cb, acc0);
    put(mb + 1u, cb, acc1);
    put(mb + 2u, cb, acc2);
    put(mb + 3u, cb, acc3);
}
"#;

/// The GRU recurrence (ONNX, linear_before_reset = 1, hidden 256): `lines` lines per
/// workgroup, unrolled so the accumulators stay in registers; each line runs for its own
/// length (the backward direction starts at its own end).
fn recurrence(lines: usize) -> String {
    let (mut init, mut acc, mut fma, mut gate, mut store) = (String::new(), String::new(), String::new(), String::new(), String::new());
    for l in 0..lines {
        init += &format!("    var len{l} = 0u;\n    if (n0 + {l}u < d.n) {{ len{l} = lens[n0 + {l}u]; }}\n");
        acc += &format!("        var az{l} = 0.0; var ar{l} = 0.0; var ah{l} = 0.0;\n");
        fma += &format!("            let h{l} = h[{l}u * 64u + q]; az{l} += dot(wz, h{l}); ar{l} += dot(wr, h{l}); ah{l} += dot(wh, h{l});\n");
        gate += &format!(
            "        let on{l} = s < len{l};
        var t{l} = s;
        if (dir == 1u) {{ t{l} = len{l} - 1u - s; }}
        var hn{l} = h[{l}u * 64u + j / 4u][j % 4u];
        if (on{l}) {{
            let gb = (t{l} * d.n + n0 + {l}u) * 1536u + dir * 768u;
            let z = sigmoid(g[gb + j] + az{l} + bz);
            let rr = sigmoid(g[gb + 256u + j] + ar{l} + br);
            let c = tanh(g[gb + 512u + j] + rr * (ah{l} + bh));
            hn{l} = (1.0 - z) * c + z * hn{l};
        }}
"
        );
        store += &format!(
            "        h[{l}u * 64u + j / 4u][j % 4u] = hn{l};\n        if (on{l}) {{ y[(t{l} * d.n + n0 + {l}u) * 512u + dir * 256u + j] = hn{l}; }}\n"
        );
    }
    format!(
        r#"
struct Dims {{ t: u32, n: u32, pad0: u32, pad1: u32 }}
@group(0) @binding(0) var<storage, read> g: array<f32>;        // [t][n][dir][768]: W x + Wb
@group(0) @binding(1) var<storage, read> r: array<vec4<f32>>;  // [dir][768][64]
@group(0) @binding(2) var<storage, read> rb: array<f32>;       // [dir][768]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // [t][n][512]
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<storage, read> lens: array<u32>;     // [n]
var<workgroup> h: array<vec4<f32>, {hsize}>;

fn sigmoid(v: f32) -> f32 {{ return 1.0 / (1.0 + exp(-v)); }}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let j = lid.x;
    let n0 = wid.x * {lines}u;
    let dir = wid.y;
    let oz = (dir * 768u + j) * 64u;
    let or = (dir * 768u + 256u + j) * 64u;
    let oh = (dir * 768u + 512u + j) * 64u;
    let bz = rb[dir * 768u + j];
    let br = rb[dir * 768u + 256u + j];
    let bh = rb[dir * 768u + 512u + j];
{init}    for (var i = j; i < {hsize}u; i += 256u) {{ h[i] = vec4<f32>(0.0); }}
    workgroupBarrier();
    for (var s = 0u; s < d.t; s++) {{
{acc}        for (var q = 0u; q < 64u; q++) {{
            let wz = r[oz + q];
            let wr = r[or + q];
            let wh = r[oh + q];
{fma}        }}
{gate}        workgroupBarrier();
{store}        workgroupBarrier();
    }}
}}
"#,
        hsize = lines * 64
    )
}

/// The recurrence with one thread per row of R (768 per workgroup): three times the loads in
/// flight per step; the gate rows meet in workgroup memory and the first 256 threads combine them.
fn recurrence768(lines: usize, f16: bool) -> String {
    // The q loop in blocks of UNROLL: all loads of a block are issued before its FMAs, so many
    // L2 round trips overlap instead of one per iteration.
    let unroll: usize = std::env::var("UNROLL").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut loads = String::new();
    for k in 0..unroll {
        loads += &if f16 { format!("            let w{k} = vec4<f32>(r[orow + q0 + {k}u]);\n") } else { format!("            let w{k} = r[orow + q0 + {k}u];\n") };
    }
    let (mut init, mut acc, mut fma, mut part, mut gate) = (String::new(), String::new(), String::new(), String::new(), String::new());
    for l in 0..lines {
        init += &format!("    var len{l} = 0u;\n    if (n0 + {l}u < d.n) {{ len{l} = lens[n0 + {l}u]; }}\n");
        acc += &format!("        var a{l} = 0.0;\n");
        let terms: Vec<String> = (0..unroll).map(|k| format!("dot(w{k}, h[{l}u * 64u + q0 + {k}u])")).collect();
        fma += &format!("            a{l} += {};\n", terms.join(" + "));
        part += &format!("        part[{l}u * 768u + j] = a{l};\n");
        gate += &format!(
            "            if (s < len{l}) {{
                var t = s;
                if (dir == 1u) {{ t = len{l} - 1u - s; }}
                let gb = (t * d.n + n0 + {l}u) * 1536u + dir * 768u;
                let z = sigmoid(g[gb + j] + part[{l}u * 768u + j] + bz);
                let rr = sigmoid(g[gb + 256u + j] + part[{l}u * 768u + 256u + j] + br);
                let c = tanh(g[gb + 512u + j] + rr * (part[{l}u * 768u + 512u + j] + bh));
                let hn = (1.0 - z) * c + z * h[{l}u * 64u + j / 4u][j % 4u];
                h[{l}u * 64u + j / 4u][j % 4u] = hn;
                y[(t * d.n + n0 + {l}u) * 512u + dir * 256u + j] = hn;
            }}
"
        );
    }
    format!(
        r#"
struct Dims {{ t: u32, n: u32, pad0: u32, pad1: u32 }}
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rb: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<storage, read> lens: array<u32>;
var<workgroup> h: array<vec4<f32>, {hsize}>;
var<workgroup> part: array<f32, {psize}>;

fn sigmoid(v: f32) -> f32 {{ return 1.0 / (1.0 + exp(-v)); }}

@compute @workgroup_size(768)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let j = lid.x;
    let n0 = wid.x * {lines}u;
    let dir = wid.y;
    let orow = (dir * 768u + j) * 64u;
    let jb = dir * 768u + (j % 256u);
    let bz = rb[jb];
    let br = rb[jb + 256u];
    let bh = rb[jb + 512u];
{init}    for (var i = j; i < {hsize}u; i += 768u) {{ h[i] = vec4<f32>(0.0); }}
    workgroupBarrier();
    for (var s = 0u; s < d.t; s++) {{
{acc}        for (var q0 = 0u; q0 < 64u; q0 += {unroll}u) {{
{loads}{fma}        }}
{part}        workgroupBarrier();
        if (j < 256u) {{
{gate}        }}
        workgroupBarrier();
    }}
}}
"#,
        hsize = lines * 64,
        psize = lines * 768
    )
}

pub(crate) fn shader(src: &str) -> String {
    src.replace("EPILOGUE_FN", EPILOGUE).replace("ENTRY", ENTRY).replace("INDEX", INDEX)
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    zeros: Buf,
    /// R of the GRU is kept in f16 (the adapter has SHADER_F16 and F16 is not "0").
    pub f16: bool,
    /// Scratch buffers by size, reused instead of created: wgpu zero-fills every new buffer
    /// before its first use.
    pool: std::cell::RefCell<HashMap<u64, Vec<Buf>>>,
    /// Bytes held in the pool, capped at `pool_cap`: varied batch shapes would otherwise keep
    /// buffers of every size ever used.
    pool_bytes: std::cell::Cell<u64>,
    pool_cap: u64,
}

/// A 4D NCHW tensor on the GPU.
#[derive(Clone)]
pub struct Tensor {
    pub buf: Buf,
    pub shape: [usize; 4],
}

impl Gpu {
    pub fn new(lines: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        let f16 = adapter.features().contains(wgpu::Features::SHADER_F16) && std::env::var("F16").map_or(true, |v| v != "0");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: if f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let mut pipes = HashMap::new();
        let mut sources: Vec<(&'static str, String)> = crate::nd::shaders();
        sources.extend([
            ("conv_fast", shader(&CONV.replace("FAST_OR_GENERIC", CONV_FAST_LOAD))),
            ("conv128", shader(CONV128)),
            ("conv64v", shader(CONV64V)),
            ("conv", shader(&CONV.replace("FAST_OR_GENERIC", CONV_GENERIC_LOAD))),
            ("depthwise", shader(DEPTHWISE)),
            ("convt", shader(CONV_TRANSPOSE)),
            ("conv_direct", shader(CONV_DIRECT)),
            ("maxpool", shader(MAXPOOL)),
            ("slice", shader(SLICE)),
            ("concat", shader(CONCAT)),
            ("activate", shader(ACTIVATE)),
            ("avgpool_t", shader(AVGPOOL_T)),
            ("grey_resize", shader(GREY_RESIZE)),
            ("mask_up", shader(MASK_UP)),
            ("gemm", GEMM.to_string()),
            ("recurrence", {
                let src = if std::env::var("RNN").is_ok_and(|v| v == "256") { recurrence(lines) } else { recurrence768(lines, f16) };
                // Diagnosis only (wrong results): NOR=1 replaces the loads of R by a constant.
                let src = if f16 {
                    format!("enable f16;\n{}", src.replace("var<storage, read> r: array<vec4<f32>>", "var<storage, read> r: array<vec4<f16>>"))
                        .replace("r[oz + q]", "vec4<f32>(r[oz + q])")
                        .replace("r[or + q]", "vec4<f32>(r[or + q])")
                        .replace("r[oh + q]", "vec4<f32>(r[oh + q])")
                        .replace("r[orow + q]", "vec4<f32>(r[orow + q])")
                } else {
                    src
                };
                if std::env::var_os("NOR").is_some() {
                    src.replace("r[oz + q]", "r[0]").replace("r[or + q]", "r[0]").replace("r[oh + q]", "r[0]").replace("r[orow + q]", "r[0]")
                } else {
                    src
                }
            }),
        ]);
        for (name, src) in sources {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.insert(name, pipe);
        }
        let zeros = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0f32; 1024]),
            usage: wgpu::BufferUsages::STORAGE,
        }));
        Ok(Self { device, queue, adapter: format!("{} ({:?}{})", info.name, info.backend, if f16 { ", R in f16" } else { "" }), pipes, zeros, f16, pool: Default::default(), pool_bytes: Default::default(), pool_cap: std::env::var("POOLMB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1024) << 20 })
    }

    pub fn upload(&self, data: &[f32]) -> Buf {
        Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(if data.is_empty() { &[0f32][..] } else { data }),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    }

    pub(crate) fn upload_u32(&self, data: &[u32]) -> Buf {
        Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    }

    pub fn storage(&self, floats: usize) -> Buf {
        let size = (floats.max(1) * 4) as u64;
        if std::env::var_os("NOPOOL").is_none()
            && let Some(b) = self.pool.borrow_mut().get_mut(&size).and_then(|v| v.pop())
        {
            self.pool_bytes.set(self.pool_bytes.get().saturating_sub(size));
            return b;
        }
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (floats.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Give scratch buffers back for reuse by later work (queue order keeps them safe).
    pub fn recycle(&self, bufs: impl IntoIterator<Item = Buf>) {
        let mut pool = self.pool.borrow_mut();
        // Only scratch buffers (storage + copy source, as `storage` makes them) are reused.
        let scratch = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        for b in bufs.into_iter().filter(|b| b.usage() == scratch) {
            if self.pool_bytes.get() + b.size() > self.pool_cap {
                continue; // dropped: wgpu frees it once the work using it is done
            }
            self.pool_bytes.set(self.pool_bytes.get() + b.size());
            pool.entry(b.size()).or_default().push(b);
        }
    }

    pub(crate) fn uniform(&self, v: &[u32]) -> Buf {
        let mut v = v.to_vec();
        while v.len() % 4 != 0 {
            v.push(0);
        }
        Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::UNIFORM,
        }))
    }

    pub(crate) fn dispatch(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], groups: (u32, u32)) {
        self.dispatch3(enc, pipe, buffers, (groups.0, groups.1, 1));
    }

    pub(crate) fn dispatch3(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], groups: (u32, u32, u32)) {
        let pipe = &self.pipes[pipe];
        let entries: Vec<wgpu::BindGroupEntry> =
            buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }

    /// Dispatch one thread per element over a 2D grid of 256-wide workgroups.
    pub(crate) fn elementwise(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], total: usize) {
        let g = total.div_ceil(256);
        self.dispatch(enc, pipe, buffers, (g.min(65535) as u32, g.div_ceil(65535) as u32));
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn submit_wait(&self, enc: wgpu::CommandEncoder) {
        self.queue.submit([enc.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Submit `enc` with a copy of `floats` values of `buf` and return them.
    pub fn read(&self, mut enc: wgpu::CommandEncoder, buf: &Buf, floats: usize) -> Vec<f32> {
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (floats * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, (floats * 4) as u64);
        self.queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let view = read.slice(..).get_mapped_range().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&view[..]).to_vec();
        drop(view);
        out
    }
}

enum Op {
    Conv { w: Buf, b: Buf, cin: usize, cout: usize, kh: usize, kw: usize, pad: usize, stride: usize, act: u32, fast: bool },
    Depthwise { w: Buf, b: Buf, kh: usize, kw: usize, pad: usize, stride: usize, act: u32 },
    /// One input channel, stride 1, with an optional fused MaxPool (kernel = stride).
    Direct { w: Buf, b: Buf, cout: usize, kh: usize, kw: usize, pad: usize, act: u32, pool: (usize, usize) },
    ConvT { w: Buf, b: Buf, cin: usize, cout: usize, kh: usize, kw: usize, stride: usize, act: u32 },
    MaxPool { kh: usize, kw: usize },
    Slice { starts: Vec<i64>, ends: Vec<i64>, axes: Vec<i64> },
    Concat,
    Act(u32),
    Identity,
}

struct Step {
    op: Op,
    inputs: Vec<String>,
    output: String,
}

pub struct Graph {
    steps: Vec<Step>,
    input: String,
    pub output: String,
}

fn act_of(op: &str) -> Option<u32> {
    match op {
        "Relu" => Some(1),
        "Sigmoid" => Some(2),
        _ => None,
    }
}

/// One value for a symmetric 2D attribute ([a, a] or [a, a, a, a]); `default` when absent.
fn symmetric(node: &Node, name: &str, default: usize) -> Result<usize, String> {
    match node.ints(name) {
        None => Ok(default),
        Some(v) if v.is_empty() => Ok(default),
        Some(v) if v.iter().all(|&x| x == v[0]) && v[0] >= 0 => Ok(v[0] as usize),
        Some(v) => Err(format!("{}: asymmetric {name} {v:?}", node.op)),
    }
}

impl Graph {
    /// The graph of `onnx` up to (not including) the first node of type `stop`, if given.
    /// Returns the graph and the stop node.
    pub fn build(gpu: &Gpu, onnx: &Onnx, stop: Option<&str>) -> Result<(Self, Option<Node>), String> {
        let mut steps = Vec::new();
        let mut i = 0;
        let mut stop_node = None;
        let nodes = &onnx.nodes;
        while i < nodes.len() {
            let n = &nodes[i];
            if Some(n.op.as_str()) == stop {
                stop_node = Some(n.clone());
                break;
            }
            let mut output = n.outputs[0].clone();
            // An activation right after, reading only this output, is fused into it.
            let mut fused = 0;
            if matches!(n.op.as_str(), "Conv" | "ConvTranspose")
                && let Some(next) = nodes.get(i + 1)
                && let Some(a) = act_of(&next.op)
                && next.inputs[0] == output
                && onnx.readers(&output) == 1
            {
                fused = a;
                output = next.outputs[0].clone();
                i += 1;
            }
            let bias = |idx: usize, count: usize| -> Result<Buf, String> {
                match n.inputs.get(idx).filter(|s| !s.is_empty()) {
                    Some(name) => Ok(gpu.upload(&onnx.weight(name)?.data)),
                    None => Ok(gpu.upload(&vec![0f32; count])),
                }
            };
            let op = match n.op.as_str() {
                "Conv" => {
                    let w = onnx.weight(&n.inputs[1])?;
                    let [cout, cin_g, kh, kw] = <[usize; 4]>::try_from(w.dims.clone()).map_err(|_| "conv weight not 4D")?;
                    let group = n.int("group").unwrap_or(1) as usize;
                    if symmetric(n, "dilations", 1)? != 1 {
                        return Err("dilated conv".into());
                    }
                    let (pad, stride) = (symmetric(n, "pads", 0)?, symmetric(n, "strides", 1)?);
                    let b = bias(2, cout)?;
                    if group == 1 && cin_g == 1 && stride == 1 && std::env::var_os("NODIRECT").is_none() {
                        // A MaxPool right after, reading only this output, is fused in.
                        let mut pool = (1, 1);
                        if let Some(next) = nodes.get(i + 1)
                            && next.op == "MaxPool"
                            && next.inputs[0] == output
                            && onnx.readers(&output) == 1
                            && let Some(k) = next.ints("kernel_shape")
                            && next.ints("strides").is_none_or(|s| s == k)
                            && symmetric(next, "pads", 0)? == 0
                            && next.int("ceil_mode").unwrap_or(0) == 0
                        {
                            pool = (k[0] as usize, k[1] as usize);
                            output = next.outputs[0].clone();
                            i += 1;
                        }
                        Op::Direct { w: gpu.upload(&w.data), b, cout, kh, kw, pad, act: fused, pool }
                    } else if group == 1 {
                        let cin = cin_g;
                        let fast = cin % 16 == 0 && std::env::var_os("NOFAST").is_none();
                        // [co][ci][ky][kx] -> [ky][kx][ci][co] (fast) or [ci][ky][kx][co].
                        let mut t = vec![0f32; w.data.len()];
                        for co in 0..cout {
                            for ci in 0..cin {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let v = w.data[((co * cin + ci) * kh + ky) * kw + kx];
                                        let k = if fast { (ky * kw + kx) * cin + ci } else { (ci * kh + ky) * kw + kx };
                                        t[k * cout + co] = v;
                                    }
                                }
                            }
                        }
                        Op::Conv { w: gpu.upload(&t), b, cin, cout, kh, kw, pad, stride, act: fused, fast }
                    } else if cin_g == 1 && group == cout {
                        Op::Depthwise { w: gpu.upload(&w.data), b, kh, kw, pad, stride, act: fused }
                    } else {
                        return Err(format!("grouped conv with group {group}"));
                    }
                }
                "ConvTranspose" => {
                    let w = onnx.weight(&n.inputs[1])?;
                    let [cin, cout, kh, kw] = <[usize; 4]>::try_from(w.dims.clone()).map_err(|_| "convT weight not 4D")?;
                    if n.int("group").unwrap_or(1) != 1 || symmetric(n, "pads", 0)? != 0 || symmetric(n, "output_padding", 0)? != 0 {
                        return Err("unsupported ConvTranspose attributes".into());
                    }
                    Op::ConvT { w: gpu.upload(&w.data), b: bias(2, cout)?, cin, cout, kh, kw, stride: symmetric(n, "strides", 1)?, act: fused }
                }
                "MaxPool" => {
                    let k = n.ints("kernel_shape").ok_or("MaxPool without kernel_shape")?;
                    let s = n.ints("strides").unwrap_or_else(|| k.clone());
                    if k != s || symmetric(n, "pads", 0)? != 0 || n.int("ceil_mode").unwrap_or(0) != 0 {
                        return Err("unsupported MaxPool attributes".into());
                    }
                    Op::MaxPool { kh: k[0] as usize, kw: k[1] as usize }
                }
                "Slice" => {
                    let get = |idx: usize| n.inputs.get(idx).and_then(|s| onnx.ints.get(s)).cloned();
                    let steps = get(4).unwrap_or_default();
                    if steps.iter().any(|&s| s != 1) {
                        return Err("Slice with steps".into());
                    }
                    Op::Slice {
                        starts: get(1).ok_or("Slice starts not constant")?,
                        ends: get(2).ok_or("Slice ends not constant")?,
                        axes: get(3).unwrap_or_else(|| (0..get(1).map_or(0, |s| s.len()) as i64).collect()),
                    }
                }
                "Concat" => {
                    if n.int("axis") != Some(1) {
                        return Err("Concat not on channels".into());
                    }
                    Op::Concat
                }
                "Pad" => {
                    let pads = n.inputs.get(1).and_then(|s| onnx.ints.get(s).cloned().or_else(|| onnx.weights.get(s).map(|w| w.data.iter().map(|&v| v as i64).collect())));
                    if pads.is_some_and(|p| p.iter().any(|&v| v != 0)) {
                        return Err("non-zero Pad".into());
                    }
                    Op::Identity
                }
                op if act_of(op).is_some() => Op::Act(act_of(op).unwrap_or(0)),
                other => return Err(format!("unsupported op {other}")),
            };
            let inputs = match op {
                Op::Concat => n.inputs.clone(),
                _ => vec![n.inputs[0].clone()],
            };
            steps.push(Step { op, inputs, output });
            i += 1;
        }
        let output = match &stop_node {
            Some(s) => s.inputs[0].clone(),
            None => onnx.output.clone(),
        };
        Ok((Self { steps, input: onnx.input.clone(), output }, stop_node))
    }

    /// Record the graph on `x`; returns the output tensor.
    pub fn run(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: Tensor) -> Result<Tensor, String> {
        let mut values: HashMap<String, Tensor> = HashMap::new();
        values.insert(self.input.clone(), x);
        let mut keep = Vec::new();
        // OPPROFILE: wait after every node and add its time to its kind (slow; diagnosis only).
        let opprofile = std::env::var_os("OPPROFILE").is_some();
        let mut kinds: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
        for s in &self.steps {
            let started = std::time::Instant::now();
            let a = values.get(&s.inputs[0]).ok_or_else(|| format!("missing value {}", s.inputs[0]))?.clone();
            let [n, c, h, w] = a.shape;
            let out = match &s.op {
                Op::Conv { w: wt, b, cin, cout, kh, kw, pad, stride, act, fast } => {
                    let (oh, ow) = ((h + 2 * pad - kh) / stride + 1, (w + 2 * pad - kw) / stride + 1);
                    let y = gpu.storage(n * cout * oh * ow);
                    let p = gpu.uniform(&[n, *cin, h, w, *cout, oh, ow, *kh, *kw, *pad, *stride, *act as usize, *pad, *stride].map(|v| v as u32));
                    let wide = *fast && std::env::var_os("NO128").is_none();
                    let vec = *fast && std::env::var_os("CONVV").is_some();
                    let (pipe, mtile) = if wide && !vec { ("conv128", 128) } else if vec { ("conv64v", 64) } else if *fast { ("conv_fast", 64) } else { ("conv", 64) };
                    gpu.dispatch(enc, pipe, &[&a.buf, wt, b, &y, &p, &a.buf], ((n * oh * ow).div_ceil(mtile) as u32, cout.div_ceil(64) as u32));
                    keep.push(p);
                    Tensor { buf: y, shape: [n, *cout, oh, ow] }
                }
                Op::Direct { w: wt, b, cout, kh, kw, pad, act, pool } => {
                    let (ch, cw) = (h + 2 * pad - kh + 1, w + 2 * pad - kw + 1);
                    let (oh, ow) = (ch / pool.0, cw / pool.1);
                    let y = gpu.storage(n * cout * oh * ow);
                    let p = gpu.uniform(&[n, h, w, *cout, oh, ow, *kh, *kw, *pad, *act as usize, pool.0, pool.1].map(|v| v as u32));
                    gpu.elementwise(enc, "conv_direct", &[&a.buf, wt, b, &y, &p], n * cout * oh * ow);
                    keep.push(p);
                    Tensor { buf: y, shape: [n, *cout, oh, ow] }
                }
                Op::Depthwise { w: wt, b, kh, kw, pad, stride, act } => {
                    let (oh, ow) = ((h + 2 * pad - kh) / stride + 1, (w + 2 * pad - kw) / stride + 1);
                    let y = gpu.storage(n * c * oh * ow);
                    let p = gpu.uniform(&[n, c, h, w, oh, ow, *kh, *kw, *pad, *stride, *act as usize, *pad, *stride].map(|v| v as u32));
                    gpu.elementwise(enc, "depthwise", &[&a.buf, wt, b, &y, &p], n * c * oh * ow);
                    keep.push(p);
                    Tensor { buf: y, shape: [n, c, oh, ow] }
                }
                Op::ConvT { w: wt, b, cin, cout, kh, kw, stride, act } => {
                    let (oh, ow) = ((h - 1) * stride + kh, (w - 1) * stride + kw);
                    let y = gpu.storage(n * cout * oh * ow);
                    let p = gpu.uniform(&[n, *cin, h, w, *cout, oh, ow, *kh, *kw, *stride, *act as usize, 0].map(|v| v as u32));
                    gpu.elementwise(enc, "convt", &[&a.buf, wt, b, &y, &p], n * cout * oh * ow);
                    keep.push(p);
                    Tensor { buf: y, shape: [n, *cout, oh, ow] }
                }
                Op::MaxPool { kh, kw } => {
                    let (oh, ow) = (h / kh, w / kw);
                    let y = gpu.storage(n * c * oh * ow);
                    let p = gpu.uniform(&[n * c, h, w, oh, ow, *kh, *kw, 0].map(|v| v as u32));
                    gpu.elementwise(enc, "maxpool", &[&a.buf, &y, &p], n * c * oh * ow);
                    keep.push(p);
                    Tensor { buf: y, shape: [n, c, oh, ow] }
                }
                Op::Slice { starts, ends, axes } => {
                    let mut lo = [0usize; 4];
                    let mut out = a.shape;
                    for ((&st, &en), &ax) in starts.iter().zip(ends).zip(axes) {
                        let ax = if ax < 0 { ax + 4 } else { ax } as usize;
                        let dim = a.shape[ax] as i64;
                        let clamp = |v: i64| (if v < 0 { v + dim } else { v }).clamp(0, dim) as usize;
                        let (s0, e0) = (clamp(st), clamp(en));
                        lo[ax] = s0;
                        out[ax] = e0.saturating_sub(s0);
                    }
                    let total: usize = out.iter().product();
                    let y = gpu.storage(total);
                    let p = gpu.uniform(&[n, c, h, w, out[0], out[1], out[2], out[3], lo[0], lo[1], lo[2], lo[3]].map(|v| v as u32));
                    gpu.elementwise(enc, "slice", &[&a.buf, &y, &p], total);
                    keep.push(p);
                    Tensor { buf: y, shape: out }
                }
                Op::Concat => {
                    let parts: Vec<Tensor> =
                        s.inputs.iter().map(|i| values.get(i).cloned().ok_or_else(|| format!("missing value {i}"))).collect::<Result<_, _>>()?;
                    let ctot: usize = parts.iter().map(|t| t.shape[1]).sum();
                    if parts.iter().any(|t| t.shape[0] != n || t.shape[2] != h || t.shape[3] != w) {
                        return Err(format!("Concat shapes differ: {:?}", parts.iter().map(|t| t.shape).collect::<Vec<_>>()));
                    }
                    let y = gpu.storage(n * ctot * h * w);
                    let mut off = 0;
                    for t in &parts {
                        let p = gpu.uniform(&[n, t.shape[1], h * w, ctot, off, 0, 0, 0].map(|v| v as u32));
                        gpu.elementwise(enc, "concat", &[&t.buf, &y, &p], n * t.shape[1] * h * w);
                        keep.push(p);
                        off += t.shape[1];
                    }
                    Tensor { buf: y, shape: [n, ctot, h, w] }
                }
                Op::Act(act) => {
                    let total = n * c * h * w;
                    let y = gpu.storage(total);
                    let p = gpu.uniform(&[total as u32, *act, 0, 0]);
                    gpu.elementwise(enc, "activate", &[&a.buf, &y, &p], total);
                    keep.push(p);
                    Tensor { buf: y, shape: a.shape }
                }
                Op::Identity => a.clone(),
            };
            if opprofile {
                let done = std::mem::replace(enc, gpu.encoder());
                gpu.submit_wait(done);
                let kind = match &s.op {
                    Op::Conv { kh, cin, fast, .. } => format!("conv {kh}x{kh} cin {} {}", if *cin >= 64 { ">=64" } else { "<64" }, if *fast { "fast" } else { "generic" }),
                    Op::Depthwise { .. } => "depthwise".into(),
                    Op::Direct { pool, .. } => format!("conv direct (pool {pool:?})"),
                    Op::ConvT { .. } => "convtranspose".into(),
                    Op::MaxPool { .. } => "maxpool".into(),
                    Op::Slice { .. } => "slice".into(),
                    Op::Concat => "concat".into(),
                    Op::Act(_) => "activation".into(),
                    Op::Identity => "identity".into(),
                };
                let e = kinds.entry(kind).or_default();
                e.0 += started.elapsed().as_secs_f64() * 1e3;
                e.1 += 1;
            }
            values.insert(s.output.clone(), out);
        }
        if opprofile {
            for (k, (t, n)) in &kinds {
                eprintln!("      {k:<32} {n:>3} x  {t:7.2} ms");
            }
        }
        let out = values.remove(&self.output).ok_or_else(|| format!("graph output {} not computed", self.output))?;
        let input = values.remove(&self.input);
        // Every other value is scratch now; the caller's input is not ours to reuse.
        let scratch: Vec<Buf> = values.into_values().map(|t| t.buf).filter(|b| !Arc::ptr_eq(b, &out.buf) && input.as_ref().is_none_or(|i| !Arc::ptr_eq(b, &i.buf))).collect();
        let mut seen = std::collections::HashSet::new();
        gpu.recycle(scratch.into_iter().filter(|b| seen.insert(Arc::as_ptr(b) as usize)));
        let _ = keep;
        Ok(out)
    }
}

pub struct Detector {
    graph: Graph,
}

impl Detector {
    pub fn new(gpu: &Gpu, onnx: &Onnx) -> Result<Self, String> {
        Ok(Self { graph: Graph::build(gpu, onnx, None)?.0 })
    }

    /// The whole detection on the GPU, as ocrs' `detect_words` does it up to the mask: the
    /// page's RGB bytes in, ocrs' grey values padded to the model size and resized to it, the
    /// model, its output cut to the unpadded part, resized back to the page and thresholded.
    /// Returns one row of `ceil(w / 32)` words per page row, a bit per pixel.
    pub fn detect_page(&self, gpu: &Gpu, rgb: &[u8], w: usize, h: usize, threshold: f32) -> Result<Vec<u32>, String> {
        let (mh, mw) = (800usize, 600usize);
        let (ph, pw) = (h.max(mh), w.max(mw));
        let mut words: Vec<u32> = vec![0; rgb.len().div_ceil(4)];
        for (k, c) in rgb.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            words[k] = u32::from_le_bytes(b);
        }
        let src = Arc::new(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        }));
        let mut enc = gpu.encoder();
        let input = gpu.storage(mh * mw);
        let f = |v: f32| v.to_bits() as usize;
        let p = gpu.uniform(&[w, h, pw, ph, mw, mh, f(pw as f32 / mw as f32), f(ph as f32 / mh as f32)].map(|v| v as u32));
        gpu.elementwise(&mut enc, "grey_resize", &[&src, &input, &p], mh * mw);
        let mask = self.graph.run(gpu, &mut enc, Tensor { buf: input.clone(), shape: [1, 1, mh, mw] })?;
        // The part of the mask that covers the page, not the padding.
        let (vh, vw) = (mh - (ph - h) * mh / ph, mw - (pw - w) * mw / pw);
        let row_words = w.div_ceil(32);
        let bits = gpu.storage(row_words * h);
        let p2 = gpu.uniform(&[mw, vw, vh, w, h, row_words, f(vw as f32 / w as f32), f(vh as f32 / h as f32), f(threshold), 0, 0, 0].map(|v| v as u32));
        gpu.elementwise(&mut enc, "mask_up", &[&mask.buf, &bits, &p2], row_words * h);
        let out = gpu.read(enc, &bits, row_words * h);
        gpu.recycle([input, mask.buf, bits]);
        Ok(out.into_iter().map(f32::to_bits).collect())
    }

    /// The text probability map for a [1, 1, h, w] input.
    pub fn detect(&self, gpu: &Gpu, input: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let mut enc = gpu.encoder();
        let out = self.graph.run(gpu, &mut enc, Tensor { buf: gpu.upload(input), shape: [1, 1, h, w] })?;
        let count = out.shape.iter().product();
        Ok(gpu.read(enc, &out.buf, count))
    }
}

struct GruLayer {
    w_t: Buf,
    wb: Buf,
    r: Buf,
    rb: Buf,
    input: usize,
}

pub struct Recognizer {
    trunk: Graph,
    pool_kh: usize,
    layers: Vec<GruLayer>,
    lin_w_t: Buf,
    lin_b: Buf,
    classes: usize,
    lines: usize,
}

impl Recognizer {
    pub fn new(gpu: &Gpu, onnx: &Onnx, lines: usize) -> Result<Self, String> {
        let (trunk, pool) = Graph::build(gpu, onnx, Some("AveragePool"))?;
        let pool = pool.ok_or("no AveragePool in the recognition model")?;
        let k = pool.ints("kernel_shape").ok_or("AveragePool without kernel_shape")?;
        if k.len() != 2 || k[1] != 1 || pool.ints("strides").as_deref() != Some(&k[..]) {
            return Err(format!("unsupported AveragePool {k:?}"));
        }
        let mut layers = Vec::new();
        for n in onnx.nodes.iter().filter(|n| n.op == "GRU") {
            if n.int("hidden_size") != Some(256) || n.int("linear_before_reset") != Some(1) || n.string("direction").as_deref() != Some("bidirectional") {
                return Err("unsupported GRU attributes".into());
            }
            let (w, r, b) = (onnx.weight(&n.inputs[1])?, onnx.weight(&n.inputs[2])?, onnx.weight(&n.inputs[3])?);
            let input = w.dims[2];
            // W [2][768][in] as rows of the projection -> [in][1536] for GEMM.
            let mut w_t = vec![0f32; w.data.len()];
            for row in 0..1536 {
                for i in 0..input {
                    w_t[i * 1536 + row] = w.data[row * input + i];
                }
            }
            let (mut wb, mut rb) = (Vec::new(), Vec::new());
            for dir in 0..2 {
                wb.extend_from_slice(&b.data[dir * 1536..dir * 1536 + 768]);
                rb.extend_from_slice(&b.data[dir * 1536 + 768..dir * 1536 + 1536]);
            }
            let r_buf = if gpu.f16 {
                let bits: Vec<u16> = r.data.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
                Arc::new(gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&bits),
                    usage: wgpu::BufferUsages::STORAGE,
                }))
            } else {
                gpu.upload(&r.data)
            };
            layers.push(GruLayer { w_t: gpu.upload(&w_t), wb: gpu.upload(&wb), r: r_buf, rb: gpu.upload(&rb), input });
        }
        if layers.len() != 2 {
            return Err(format!("{} GRU layers", layers.len()));
        }
        let matmul = onnx.nodes.iter().find(|n| n.op == "MatMul").ok_or("no MatMul")?;
        let lin = onnx.weight(&matmul.inputs[1])?; // [512][classes]: already [in][out]
        let add = onnx.nodes.iter().find(|n| n.op == "Add" && n.inputs.contains(&matmul.outputs[0])).ok_or("no bias Add")?;
        let bias = add.inputs.iter().find_map(|i| onnx.weights.get(i)).ok_or("no bias weight")?;
        Ok(Self {
            trunk,
            pool_kh: k[0] as usize,
            layers,
            lin_w_t: gpu.upload(&lin.data),
            lin_b: gpu.upload(&bias.data),
            classes: lin.dims[1],
            lines,
        })
    }

    /// Recognize `batches` of [n, 1, 64, w] line images (each batch its own width). Returns the
    /// log-probabilities [t_max][lines][classes] and each line's sequence length.
    pub fn recognize(&self, gpu: &Gpu, batches: &[(Vec<f32>, usize, usize)]) -> Result<(Vec<f32>, usize, Vec<usize>), String> {
        // The trunks first, one submission each, so their activations are freed in between
        // (SYNC=0: all queued without waiting).
        let profile = std::env::var_os("PROFILE").is_some();
        let sync = std::env::var("SYNC").map_or(true, |v| v != "0");
        let started = std::time::Instant::now();
        let mut trunks = Vec::new();
        for (data, n, w) in batches {
            let mut enc = gpu.encoder();
            let out = self.trunk.run(gpu, &mut enc, Tensor { buf: gpu.upload(data), shape: [*n, 1, 64, *w] })?;
            if sync {
                gpu.submit_wait(enc);
            } else {
                gpu.queue.submit([enc.finish()]);
            }
            trunks.push(out);
        }
        if profile {
            gpu.submit_wait(gpu.encoder());
            eprintln!("    trunks ({} batches): {:.1} ms", batches.len(), started.elapsed().as_secs_f64() * 1e3);
        }
        let started = std::time::Instant::now();
        let ntot: usize = batches.iter().map(|b| b.1).sum();
        let tmax = trunks.iter().map(|t| t.shape[3]).max().unwrap_or(0);
        let c = trunks.first().map_or(0, |t| t.shape[1]);
        if c != self.layers[0].input {
            return Err(format!("trunk gives {c} features, GRU takes {}", self.layers[0].input));
        }
        let mut enc = gpu.encoder();
        let x = gpu.storage(tmax * ntot * c);
        let x_keep = x.clone();
        let mut lens = Vec::with_capacity(ntot);
        let mut keep = Vec::new();
        let mut noff = 0;
        for t in &trunks {
            let [n, c, h, w] = t.shape;
            if (h - self.pool_kh) / self.pool_kh + 1 != 1 {
                return Err(format!("trunk height {h} does not pool to one row"));
            }
            let p = gpu.uniform(&[n, c, h, w, self.pool_kh, ntot, noff, 0].map(|v| v as u32));
            gpu.elementwise(&mut enc, "avgpool_t", &[&t.buf, &x, &p], n * c * w);
            keep.push(p);
            lens.extend(std::iter::repeat_n(w, n));
            noff += n;
        }
        let lens_buf = gpu.upload_u32(&lens.iter().map(|&l| l as u32).collect::<Vec<_>>());
        let mut mark = |enc: &mut wgpu::CommandEncoder, what: &str| {
            if profile {
                let done = std::mem::replace(enc, gpu.encoder());
                gpu.submit_wait(done);
                eprintln!("      {what}: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
            }
        };
        mark(&mut enc, "avgpool");
        let m = tmax * ntot;
        let gates = gpu.storage(m * 1536);
        let gates_keep = gates.clone();
        let mut outs = Vec::new();
        let mut input = x;
        for layer in &self.layers {
            let out = gpu.storage(m * 512);
            outs.push(out.clone());
            let d = gpu.uniform(&[m as u32, 1536, layer.input as u32, 0]);
            gpu.dispatch(&mut enc, "gemm", &[&input, &layer.w_t, &layer.wb, &gates, &d], (m.div_ceil(64) as u32, 1536u32.div_ceil(64)));
            mark(&mut enc, "projection");
            let d2 = gpu.uniform(&[tmax as u32, ntot as u32, 0, 0]);
            gpu.dispatch(&mut enc, "recurrence", &[&gates, &layer.r, &layer.rb, &out, &d2, &lens_buf], (ntot.div_ceil(self.lines) as u32, 2));
            mark(&mut enc, "recurrence");
            keep.extend([d, d2]);
            input = out;
        }
        let logits = gpu.storage(m * self.classes);
        let d = gpu.uniform(&[m as u32, self.classes as u32, 512, 0]);
        gpu.dispatch(&mut enc, "gemm", &[&input, &self.lin_w_t, &self.lin_b, &logits, &d], (m.div_ceil(64) as u32, self.classes.div_ceil(64) as u32));
        mark(&mut enc, "output layer");
        let mut out = gpu.read(enc, &logits, m * self.classes);
        gpu.recycle(trunks.into_iter().map(|t| t.buf).chain([x_keep, gates_keep, logits]).chain(outs));
        if profile {
            eprintln!("      readback: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
        if profile {
            eprintln!("    GRU + output over {ntot} lines, {tmax} steps: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
        // Greedy decoding needs only the arg-max, which log-softmax keeps: LOGSOFTMAX=1 for the
        // log-probabilities themselves.
        for row in out.chunks_exact_mut(self.classes).filter(|_| std::env::var_os("LOGSOFTMAX").is_some()) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ln = row.iter().map(|v| (v - max).exp()).sum::<f32>().ln() + max;
            row.iter_mut().for_each(|v| *v -= ln);
        }
        let _ = self.trunk.steps.len();
        Ok((out, tmax, lens))
    }
}

/// Keep the `zeros` buffer referenced (bias of convolutions without one uses a fresh buffer).
impl Gpu {
    pub fn zeros(&self) -> &Buf {
        &self.zeros
    }
}
