//! A general ONNX graph executor on the GPU for N-dimensional float tensors, enough for the
//! PP-OCR recognition models (CNN backbone, SVTR transformer neck, CTC head).
//!
//! Shapes are resolved for each input as the graph runs: integer "shape" subgraphs (Shape,
//! Gather, Slice, Squeeze, Unsqueeze, Concat, Cast on integers) are computed on the CPU, float
//! nodes become dispatches. Identity nodes are removed up front; a constant bias Add and an
//! activation right after a Conv or MatMul are fused into it, and BatchNormalization becomes a
//! per-channel scale and shift.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::{Gpu, shader};
use crate::onnx::{Node, Onnx};

type Buf = Arc<wgpu::Buffer>;

/// NDPROFILE: time per node kind, summed over runs (diagnosis; waits after every node).
static PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (f64, usize)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The profile so far, slowest first.
pub fn profile() -> Vec<(String, f64, usize)> {
    let p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<_> = p.iter().map(|(k, (t, n))| (k.clone(), *t, *n)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

/// The most dimensions a tensor may have here.
const RANK: usize = 6;

const BINARY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;  // total, rank, op, 0, dims[6], astr[6], bstr[6]
ENTRY
    INDEX
    if (i >= p[0]) { return; }
    var rem = i;
    var ai = 0u;
    var bi = 0u;
    for (var k = 0u; k < p[1]; k++) {
        let d = p[1] - 1u - k;
        let dim = p[4u + d];
        let c = rem % dim;
        rem = rem / dim;
        ai += c * p[10u + d];
        bi += c * p[16u + d];
    }
    let x = a[ai];
    let z = b[bi];
    var v = 0.0;
    switch p[2] {
        case 0u: { v = x + z; }
        case 1u: { v = x - z; }
        case 2u: { v = x * z; }
        case 3u: { v = x / z; }
        default: {
            if (z == 2.0) { v = x * x; } else if (z == 0.5) { v = sqrt(x); } else { v = pow(x, z); }
        }
    }
    y[i] = v;
}
"#;

const UNARY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // total, op, alpha bits, beta bits
fn erf(v: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7.
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-a * a));
}
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let v = x[i];
    let alpha = bitcast<f32>(p.z);
    let beta = bitcast<f32>(p.w);
    var r = v;
    switch p.y {
        case 1u: { r = max(v, 0.0); }
        case 2u: { r = 1.0 / (1.0 + exp(-v)); }
        case 3u: { r = clamp(alpha * v + beta, 0.0, 1.0); }
        case 4u: { r = erf(v); }
        case 5u: { r = sqrt(v); }
        default: {}
    }
    y[i] = r;
}
"#;

/// Mean over the trailing `inner` values of each of `outer` rows.
const REDUCE_MEAN: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // outer, inner
ENTRY
    INDEX
    if (i >= p.x) { return; }
    var s = 0.0;
    for (var j = 0u; j < p.y; j++) { s += x[i * p.y + j]; }
    y[i] = s / f32(p.y);
}
"#;

const SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // rows, row length
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let base = i * p.y;
    var m = -3.4e38;
    for (var j = 0u; j < p.y; j++) { m = max(m, x[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.y; j++) { s += exp(x[base + j] - m); }
    for (var j = 0u; j < p.y; j++) { y[base + j] = exp(x[base + j] - m) / s; }
}
"#;

/// y[obase + sum c_d os_d] = x[ibase + sum c_d is_d] over the index space `dims`: Transpose,
/// Slice and Concat are all this.
const COPY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;  // total, rank, ibase, obase, dims[6], is[6], os[6]
ENTRY
    INDEX
    if (i >= p[0]) { return; }
    var rem = i;
    var xi = p[2];
    var yi = p[3];
    for (var k = 0u; k < p[1]; k++) {
        let d = p[1] - 1u - k;
        let dim = p[4u + d];
        let c = rem % dim;
        rem = rem / dim;
        xi += c * p[10u + d];
        yi += c * p[16u + d];
    }
    y[yi] = x[xi];
}
"#;

/// Max or average pooling with stride and leading padding (padding never counts).
const POOL: &str = r#"
struct P { nc: u32, h: u32, w: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32, sw: u32, ph: u32, pw: u32, avg: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.nc * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let oy = (i / p.ow) % p.oh;
    let c = i / (p.ow * p.oh);
    var m = -3.4e38;
    var s = 0.0;
    var n = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        let iy = i32(oy * p.sh + ky) - i32(p.ph);
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0u; kx < p.kw; kx++) {
            let ix = i32(ox * p.sw + kx) - i32(p.pw);
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            let v = x[(c * p.h + u32(iy)) * p.w + u32(ix)];
            m = max(m, v);
            s += v;
            n += 1.0;
        }
    }
    if (p.avg == 1u) { y[i] = s / max(n, 1.0); } else { y[i] = m; }
}
"#;

/// y = x * scale[c] + shift[c], c the axis-1 index (BatchNormalization folded).
const CHANNEL_AFFINE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<f32>;  // scale[c] then shift[c]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;  // total, channels, inner
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let c = (i / p.z) % p.y;
    y[i] = x[i] * st[c] + st[p.y + c];
}
"#;

/// Batched y[b][m, k] = act(x[b][m, :] . w[b][:, k] + bias[k]), batch strides in elements (0 for
/// a shared weight). 64 x 64 tile, 4 x 4 per thread.
const BGEMM: &str = r#"
struct D { m: u32, k: u32, i: u32, act: u32, xs: u32, ws: u32, ys: u32, x0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: D;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
EPILOGUE_FN

fn put(base: u32, m: u32, c0: u32, v: vec4<f32>) {
    if (m >= d.m) { return; }
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < d.k) { y[base + m * d.k + c] = activate(v[j] + b[c], d.act); }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 64u;
    let c0 = wid.y * 64u;
    let xb = wid.z * d.xs;
    let wb = wid.z * d.ws;
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
            if (m0 + am < d.m && k0 + ak < d.i) { v = x[xb + (m0 + am) * d.i + k0 + ak]; }
            As[ak * 64u + am] = v;
            let bk = idx / 64u;
            let bc = idx % 64u;
            var wv = 0.0;
            if (k0 + bk < d.i && c0 + bc < d.k) { wv = wt[wb + (k0 + bk) * d.k + c0 + bc]; }
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
    let base = wid.z * d.ys;
    let mb = m0 + lid.x * 4u;
    let cb = c0 + lid.y * 4u;
    put(base, mb, cb, acc0);
    put(base, mb + 1u, cb, acc1);
    put(base, mb + 2u, cb, acc2);
    put(base, mb + 3u, cb, acc3);
}
"#;

/// The arg-max of each row: one workgroup per row.
const ARGMAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // rows, row length
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let row = wid.y * nw.x + wid.x;
    let t = lid.x;
    var best = -3.4e38;
    var at = 0u;
    if (row < p.x) {
        for (var j = t; j < p.y; j += 256u) {
            let v = x[row * p.y + j];
            if (v > best) { best = v; at = j; }
        }
    }
    bv[t] = best;
    bi[t] = at;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s / 2u) {
        if (t < s && (bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t]))) {
            bv[t] = bv[t + s];
            bi[t] = bi[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u && row < p.x) { y[row] = bi[0]; }
}
"#;

const EPILOGUE: &str = r#"
fn erf_approx(v: f32) -> f32 {
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

pub(crate) fn shaders() -> Vec<(&'static str, String)> {
    vec![
        ("nd_binary", shader(BINARY)),
        ("nd_unary", shader(UNARY)),
        ("nd_reduce_mean", shader(REDUCE_MEAN)),
        ("nd_softmax", shader(SOFTMAX)),
        ("nd_copy", shader(COPY)),
        ("nd_pool", shader(POOL)),
        ("nd_affine", shader(CHANNEL_AFFINE)),
        ("nd_bgemm", BGEMM.replace("EPILOGUE_FN", EPILOGUE)),
        ("nd_argmax", ARGMAX.to_string()),
    ]
}

/// A float tensor on the GPU.
#[derive(Clone)]
pub struct Tensor {
    pub buf: Buf,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
enum Value {
    F(Tensor),
    /// An integer tensor kept on the CPU (shapes and indices); its own shape.
    I(Vec<i64>, Vec<usize>),
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn axis(a: i64, rank: usize) -> Result<usize, String> {
    let r = if a < 0 { a + rank as i64 } else { a };
    if r < 0 || r as usize >= rank.max(1) {
        return Err(format!("axis {a} out of range for rank {rank}"));
    }
    Ok(r as usize)
}

/// A Conv with its weights laid out for the kernel it uses.
struct ConvW {
    w: Buf,
    b: Buf,
    cout: usize,
    cin_g: usize,
    group: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    pads: Option<[usize; 4]>,
    same_upper: bool,
    fast: bool,
}

/// What a fused node does.
enum Fused {
    /// The conv, its activation, and a tensor added after it (a residual).
    Conv(ConvW, u32, Option<String>),
    /// MatMul with a constant [K, N] weight, its bias, its activation.
    Linear { w: Buf, b: Buf, k: usize, n: usize, act: u32 },
    Affine { st: Buf },
}

pub struct NdGraph {
    nodes: Vec<Node>,
    /// Float initializers on the GPU, with their shapes.
    weights: HashMap<String, Tensor>,
    ints: HashMap<String, Vec<i64>>,
    /// Per node index: a fused implementation, and the nodes it swallowed.
    fused: HashMap<usize, Fused>,
    skip: Vec<bool>,
    input: String,
    /// The value the run returns: the graph output, or the input of a final Softmax.
    output: String,
}

impl NdGraph {
    /// `stop_before_softmax`: return the logits a final Softmax would read (arg-max needs no
    /// Softmax, and the logits are what is compared).
    pub fn new(gpu: &Gpu, onnx: &Onnx, stop_before_softmax: bool) -> Result<Self, String> {
        // Identity nodes: every reader reads their input instead.
        let mut alias: HashMap<String, String> = HashMap::new();
        for n in &onnx.nodes {
            if n.op == "Identity" {
                let src = alias.get(&n.inputs[0]).cloned().unwrap_or_else(|| n.inputs[0].clone());
                alias.insert(n.outputs[0].clone(), src);
            }
        }
        let resolve = |s: &String| alias.get(s).cloned().unwrap_or_else(|| s.clone());
        let mut nodes: Vec<Node> = onnx
            .nodes
            .iter()
            .filter(|n| n.op != "Identity")
            .map(|n| {
                let mut n = n.clone();
                n.inputs = n.inputs.iter().map(resolve).collect();
                n
            })
            .collect();
        let mut output = resolve(&onnx.output);
        if stop_before_softmax
            && let Some(last) = nodes.iter().rposition(|n| n.outputs.first() == Some(&output))
            && nodes[last].op == "Softmax"
        {
            output = nodes[last].inputs[0].clone();
            nodes.truncate(last);
        }
        let mut weights = HashMap::new();
        for (name, w) in &onnx.weights {
            weights.insert(name.clone(), Tensor { buf: gpu.upload(&w.data), shape: w.dims.clone() });
        }
        let scalar = |name: &str| onnx.weights.get(name).filter(|w| w.data.len() == 1).map(|w| w.data[0]);
        let readers = |name: &str| nodes.iter().filter(|n| n.inputs.iter().any(|i| i == name)).count() + usize::from(name == output);
        let (mut fused, mut skip) = (HashMap::new(), vec![false; nodes.len()]);
        // (node, new output): a fused node writes its last swallowed node's output.
        let mut renames: Vec<(usize, String)> = Vec::new();
        let mut i = 0;
        while i < nodes.len() {
            let n = &nodes[i];
            // A following Add of a constant bias, then an activation, each read only here.
            let mut j = i + 1;
            let mut out = n.outputs[0].clone();
            let mut bias: Option<String> = None;
            let mut act = 0;
            let takes_bias = |next: &Node, out: &str, want: usize| -> Option<String> {
                if next.op != "Add" || readers(out) != 1 {
                    return None;
                }
                let other = if next.inputs[0] == out { &next.inputs[1] } else if next.inputs[1] == out { &next.inputs[0] } else { return None };
                let w = onnx.weights.get(other)?;
                (w.data.len() == want && w.dims.iter().filter(|&&d| d != 1).count() <= 1).then(|| other.clone())
            };
            match n.op.as_str() {
                "Conv" => {
                    let w = onnx.weight(&n.inputs[1])?;
                    let [cout, cin_g, kh, kw] = <[usize; 4]>::try_from(w.dims.clone()).map_err(|_| "conv weight not 4D")?;
                    let mut b = n.inputs.get(2).filter(|s| !s.is_empty()).and_then(|s| onnx.weights.get(s)).map(|w| w.data.clone()).unwrap_or_else(|| vec![0.0; cout]);
                    if let Some(next) = nodes.get(j)
                        && let Some(name) = takes_bias(next, &out, cout)
                    {
                        for (x, y) in b.iter_mut().zip(&onnx.weight(&name)?.data) {
                            *x += y;
                        }
                        out = next.outputs[0].clone();
                        bias = Some(name);
                        j += 1;
                    }
                    if let Some(next) = nodes.get(j)
                        && next.op == "Relu"
                        && next.inputs[0] == out
                        && readers(&out) == 1
                    {
                        act = 1;
                        out = next.outputs[0].clone();
                        j += 1;
                    }
                    let _ = bias;
                    // GELU as the exporter writes it: x * (erf(x / sqrt 2) + 1) * 0.5.
                    if act == 0
                        && let (Some(div), Some(erf), Some(add), Some(mul), Some(half)) = (nodes.get(j), nodes.get(j + 1), nodes.get(j + 2), nodes.get(j + 3), nodes.get(j + 4))
                        && div.op == "Div"
                        && div.inputs[0] == out
                        && scalar(&div.inputs[1]).is_some_and(|c| (c - std::f32::consts::SQRT_2).abs() < 1e-4)
                        && erf.op == "Erf"
                        && erf.inputs[0] == div.outputs[0]
                        && add.op == "Add"
                        && add.inputs.contains(&erf.outputs[0])
                        && add.inputs.iter().any(|i| scalar(i).is_some_and(|c| (c - 1.0).abs() < 1e-6))
                        && mul.op == "Mul"
                        && mul.inputs.contains(&out)
                        && mul.inputs.contains(&add.outputs[0])
                        && half.op == "Mul"
                        && half.inputs.contains(&mul.outputs[0])
                        && half.inputs.iter().any(|i| scalar(i).is_some_and(|c| (c - 0.5).abs() < 1e-6))
                        && readers(&out) == 2
                        && [&div.outputs[0], &erf.outputs[0], &add.outputs[0], &mul.outputs[0]].iter().all(|o| readers(o) == 1)
                    {
                        act = 3;
                        out = half.outputs[0].clone();
                        j += 5;
                    }
                    // A residual Add of another computed tensor of the same shape.
                    let mut residual = None;
                    if let Some(next) = nodes.get(j)
                        && next.op == "Add"
                        && readers(&out) == 1
                        && next.inputs.contains(&out)
                        && let Some(other) = next.inputs.iter().find(|i| **i != out)
                        && !onnx.weights.contains_key(other)
                        && !onnx.ints.contains_key(other)
                    {
                        residual = Some(other.clone());
                        out = next.outputs[0].clone();
                        j += 1;
                    }
                    let group = n.int("group").unwrap_or(1) as usize;
                    let strides = n.ints("strides").unwrap_or_else(|| vec![1, 1]);
                    if n.ints("dilations").is_some_and(|d| d.iter().any(|&x| x != 1)) {
                        return Err("dilated conv".into());
                    }
                    let auto = n.string("auto_pad").unwrap_or_default();
                    let pads = n.ints("pads").map(|p| [p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize]);
                    let fast = group == 1 && cin_g % 16 == 0;
                    let data = if group == 1 {
                        let mut t = vec![0f32; w.data.len()];
                        for co in 0..cout {
                            for ci in 0..cin_g {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let k = if fast { (ky * kw + kx) * cin_g + ci } else { (ci * kh + ky) * kw + kx };
                                        t[k * cout + co] = w.data[((co * cin_g + ci) * kh + ky) * kw + kx];
                                    }
                                }
                            }
                        }
                        t
                    } else {
                        w.data.clone()
                    };
                    let conv = ConvW {
                        w: gpu.upload(&data),
                        b: gpu.upload(&b),
                        cout,
                        cin_g,
                        group,
                        kh,
                        kw,
                        stride: [strides[0] as usize, strides[1] as usize],
                        pads,
                        same_upper: auto == "SAME_UPPER",
                        fast,
                    };
                    fused.insert(i, Fused::Conv(conv, act, residual));
                }
                "MatMul" if onnx.weights.get(&n.inputs[1]).is_some_and(|w| w.dims.len() == 2) => {
                    let w = onnx.weight(&n.inputs[1])?;
                    let (k, nn) = (w.dims[0], w.dims[1]);
                    let mut b = vec![0.0; nn];
                    if let Some(next) = nodes.get(j)
                        && let Some(name) = takes_bias(next, &out, nn)
                    {
                        b.copy_from_slice(&onnx.weight(&name)?.data);
                        out = next.outputs[0].clone();
                        j += 1;
                    }
                    // SiLU: x * sigmoid(x).
                    let mut act = 0;
                    if let (Some(sig), Some(mul)) = (nodes.get(j), nodes.get(j + 1))
                        && sig.op == "Sigmoid"
                        && sig.inputs[0] == out
                        && mul.op == "Mul"
                        && mul.inputs.contains(&out)
                        && mul.inputs.contains(&sig.outputs[0])
                        && readers(&out) == 2
                        && readers(&sig.outputs[0]) == 1
                    {
                        act = 4;
                        out = mul.outputs[0].clone();
                        j += 2;
                    }
                    fused.insert(i, Fused::Linear { w: gpu.upload(&w.data), b: gpu.upload(&b), k, n: nn, act });
                }
                "BatchNormalization" => {
                    let g = |k: usize| onnx.weight(&n.inputs[k]).map(|w| w.data.clone());
                    let (scale, shift, mean, var) = (g(1)?, g(2)?, g(3)?, g(4)?);
                    let eps = n.attr_f32("epsilon").unwrap_or(1e-5);
                    let c = scale.len();
                    let mut st = vec![0f32; 2 * c];
                    for ch in 0..c {
                        let s = scale[ch] / (var[ch] + eps).sqrt();
                        st[ch] = s;
                        st[c + ch] = shift[ch] - mean[ch] * s;
                    }
                    fused.insert(i, Fused::Affine { st: gpu.upload(&st) });
                }
                _ => {}
            }
            for k in i + 1..j {
                skip[k] = true;
            }
            if j > i + 1 {
                renames.push((i, out.clone()));
            }
            i = j;
        }
        for (i, out) in renames {
            nodes[i].outputs[0] = out;
        }
        let input = onnx.input.clone();
        Ok(Self { nodes, weights, ints: onnx.ints.clone(), fused, skip, input, output })
    }

    fn get(&self, values: &HashMap<String, Value>, name: &str) -> Result<Value, String> {
        if let Some(v) = values.get(name) {
            return Ok(v.clone());
        }
        if let Some(t) = self.weights.get(name) {
            return Ok(Value::F(t.clone()));
        }
        if let Some(i) = self.ints.get(name) {
            return Ok(Value::I(i.clone(), if i.len() == 1 { vec![1] } else { vec![i.len()] }));
        }
        Err(format!("missing value {name}"))
    }

    /// Record the graph on `x`; returns the output tensor (the logits, if stopped before a
    /// final Softmax). Scratch buffers go back to the pool after the caller submits.
    pub fn run(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: Tensor, scratch: &mut Vec<Buf>) -> Result<Tensor, String> {
        let mut values: HashMap<String, Value> = HashMap::new();
        values.insert(self.input.clone(), Value::F(x));
        let mut keep = Vec::new();
        let new = |shape: &[usize], scratch: &mut Vec<Buf>| {
            let b = gpu.storage(shape.iter().product::<usize>().max(1));
            scratch.push(b.clone());
            Tensor { buf: b, shape: shape.to_vec() }
        };
        for (idx, n) in self.nodes.iter().enumerate() {
            if self.skip[idx] {
                continue;
            }
            let node_start = std::time::Instant::now();
            let arg = |k: usize| self.get(&values, &n.inputs[k]);
            let float = |k: usize| -> Result<Tensor, String> {
                match arg(k)? {
                    Value::F(t) => Ok(t),
                    Value::I(..) => Err(format!("{}: input {k} is not float", n.op)),
                }
            };
            let ints = |k: usize| -> Result<Vec<i64>, String> {
                match arg(k)? {
                    Value::I(v, _) => Ok(v),
                    Value::F(_) => Err(format!("{}: input {k} is not an integer constant", n.op)),
                }
            };
            let out: Value = match (n.op.as_str(), self.fused.get(&idx)) {
                ("Conv", Some(Fused::Conv(c, act, res))) => {
                    let r = match res {
                        Some(name) => match self.get(&values, name)? {
                            Value::F(t) => Some(t),
                            Value::I(..) => return Err("integer residual".into()),
                        },
                        None => None,
                    };
                    Value::F(self.conv(gpu, enc, &float(0)?, c, *act, r.as_ref(), scratch, &mut keep)?)
                }
                ("MatMul", Some(Fused::Linear { w, b, k, n: nn, act })) => {
                    let a = float(0)?;
                    if a.shape.last() != Some(k) {
                        return Err(format!("MatMul: {:?} by [{k}, {nn}]", a.shape));
                    }
                    let m: usize = a.shape[..a.shape.len() - 1].iter().product();
                    let mut shape = a.shape.clone();
                    *shape.last_mut().unwrap_or(&mut 0) = *nn;
                    let y = new(&shape, scratch);
                    self.gemm(gpu, enc, &a.buf, w, b, &y.buf, [m, *nn, *k, 1], [0, 0, 0], *act, &mut keep);
                    Value::F(y)
                }
                ("MatMul", None) => {
                    let (a, bm) = (float(0)?, float(1)?);
                    let r = a.shape.len();
                    if r < 2 || bm.shape.len() != r || a.shape[..r - 2] != bm.shape[..r - 2] || a.shape[r - 1] != bm.shape[r - 2] {
                        return Err(format!("MatMul shapes {:?} x {:?}", a.shape, bm.shape));
                    }
                    let (m, k, nn) = (a.shape[r - 2], a.shape[r - 1], bm.shape[r - 1]);
                    let batch: usize = a.shape[..r - 2].iter().product();
                    let mut shape = a.shape.clone();
                    shape[r - 1] = nn;
                    let y = new(&shape, scratch);
                    let zeros = gpu.upload(&vec![0f32; nn]);
                    keep.push(zeros.clone());
                    self.gemm(gpu, enc, &a.buf, &bm.buf, &zeros, &y.buf, [m, nn, k, batch], [m * k, k * nn, m * nn], 0, &mut keep);
                    Value::F(y)
                }
                ("BatchNormalization", Some(Fused::Affine { st })) => {
                    let a = float(0)?;
                    let c = a.shape.get(1).copied().unwrap_or(1);
                    let inner: usize = a.shape.iter().skip(2).product();
                    let y = new(&a.shape, scratch);
                    let total = a.shape.iter().product::<usize>();
                    let p = gpu.uniform(&[total as u32, c as u32, inner.max(1) as u32, 0]);
                    gpu.elementwise(enc, "nd_affine", &[&a.buf, st, &y.buf, &p], total);
                    keep.push(p);
                    Value::F(y)
                }
                ("Add" | "Sub" | "Mul" | "Div" | "Pow", _) => {
                    let (a, b) = (arg(0)?, arg(1)?);
                    match (a, b) {
                        (Value::F(a), Value::F(b)) => Value::F(self.binary(gpu, enc, &n.op, &a, &b, scratch, &mut keep)?),
                        (Value::I(a, sa), Value::I(b, _)) => {
                            let f = |x: i64, y: i64| match n.op.as_str() {
                                "Add" => x + y,
                                "Sub" => x - y,
                                "Mul" => x * y,
                                "Div" => if y == 0 { 0 } else { x / y },
                                _ => x,
                            };
                            let v: Vec<i64> = if b.len() == 1 { a.iter().map(|&x| f(x, b[0])).collect() } else { a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect() };
                            Value::I(v, sa)
                        }
                        _ => return Err(format!("{}: mixed integer and float inputs", n.op)),
                    }
                }
                ("Relu" | "Sigmoid" | "HardSigmoid" | "Erf" | "Sqrt", _) => {
                    let a = float(0)?;
                    let op = match n.op.as_str() {
                        "Relu" => 1,
                        "Sigmoid" => 2,
                        "HardSigmoid" => 3,
                        "Erf" => 4,
                        _ => 5,
                    };
                    let (alpha, beta) = (n.attr_f32("alpha").unwrap_or(0.2), n.attr_f32("beta").unwrap_or(0.5));
                    let y = new(&a.shape, scratch);
                    let total = a.shape.iter().product::<usize>();
                    let p = gpu.uniform(&[total as u32, op, alpha.to_bits(), beta.to_bits()]);
                    gpu.elementwise(enc, "nd_unary", &[&a.buf, &y.buf, &p], total);
                    keep.push(p);
                    Value::F(y)
                }
                ("ReduceMean", _) => {
                    let a = float(0)?;
                    let r = a.shape.len();
                    let mut axes: Vec<usize> = n.ints("axes").unwrap_or_else(|| (0..r as i64).collect()).iter().map(|&x| axis(x, r)).collect::<Result<_, _>>()?;
                    axes.sort_unstable();
                    if axes.iter().enumerate().any(|(k, &x)| x != r - axes.len() + k) {
                        return Err(format!("ReduceMean over non-trailing axes {axes:?}"));
                    }
                    let inner: usize = axes.iter().map(|&x| a.shape[x]).product();
                    let outer = a.shape.iter().product::<usize>() / inner.max(1);
                    let keepdims = n.int("keepdims").unwrap_or(1) == 1;
                    let shape: Vec<usize> = a.shape.iter().enumerate().filter_map(|(d, &s)| if axes.contains(&d) { keepdims.then_some(1) } else { Some(s) }).collect();
                    let y = new(&shape, scratch);
                    let p = gpu.uniform(&[outer as u32, inner as u32, 0, 0]);
                    gpu.elementwise(enc, "nd_reduce_mean", &[&a.buf, &y.buf, &p], outer);
                    keep.push(p);
                    Value::F(y)
                }
                ("Softmax", _) => {
                    let a = float(0)?;
                    let ax = axis(n.int("axis").unwrap_or(-1), a.shape.len())?;
                    let len: usize = a.shape[ax..].iter().product();
                    let rows = a.shape.iter().product::<usize>() / len.max(1);
                    let y = new(&a.shape, scratch);
                    let p = gpu.uniform(&[rows as u32, len as u32, 0, 0]);
                    gpu.elementwise(enc, "nd_softmax", &[&a.buf, &y.buf, &p], rows);
                    keep.push(p);
                    Value::F(y)
                }
                ("Transpose", _) => {
                    let a = float(0)?;
                    let r = a.shape.len();
                    let perm: Vec<usize> = n.ints("perm").unwrap_or_else(|| (0..r as i64).rev().collect()).iter().map(|&x| x as usize).collect();
                    let shape: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
                    let is = strides(&a.shape);
                    let y = new(&shape, scratch);
                    let src: Vec<usize> = perm.iter().map(|&p| is[p]).collect();
                    self.copy(gpu, enc, &a.buf, &y.buf, &shape, &src, &strides(&shape), 0, 0, &mut keep)?;
                    Value::F(y)
                }
                ("Slice", _) => {
                    let (starts, ends) = (ints(1)?, ints(2)?);
                    let data = arg(0)?;
                    let shape_of = |v: &Value| match v {
                        Value::F(t) => t.shape.clone(),
                        Value::I(_, s) => s.clone(),
                    };
                    let shape = shape_of(&data);
                    let r = shape.len();
                    let axes = if n.inputs.len() > 3 && !n.inputs[3].is_empty() { ints(3)? } else { (0..starts.len() as i64).collect() };
                    if n.inputs.len() > 4 && !n.inputs[4].is_empty() && ints(4)?.iter().any(|&s| s != 1) {
                        return Err("Slice with steps".into());
                    }
                    let mut lo = vec![0usize; r];
                    let mut out = shape.clone();
                    for ((&st, &en), &ax) in starts.iter().zip(&ends).zip(&axes) {
                        let ax = axis(ax, r)?;
                        let dim = shape[ax] as i64;
                        let clampv = |v: i64| (if v < 0 { v + dim } else { v }).clamp(0, dim) as usize;
                        let (s0, e0) = (clampv(st), clampv(en));
                        lo[ax] = s0;
                        out[ax] = e0.saturating_sub(s0);
                    }
                    match data {
                        Value::I(v, _) => {
                            // A 1-D integer slice (shape arithmetic).
                            Value::I(v.get(lo[0]..lo[0] + out[0]).map(<[i64]>::to_vec).unwrap_or_default(), out)
                        }
                        Value::F(a) => {
                            let is = strides(&a.shape);
                            let base: usize = lo.iter().zip(&is).map(|(l, s)| l * s).sum();
                            let y = new(&out, scratch);
                            self.copy(gpu, enc, &a.buf, &y.buf, &out, &is, &strides(&out), base, 0, &mut keep)?;
                            Value::F(y)
                        }
                    }
                }
                ("Concat", _) => {
                    let parts: Vec<Value> = (0..n.inputs.len()).map(arg).collect::<Result<_, _>>()?;
                    if parts.iter().all(|p| matches!(p, Value::I(..))) {
                        let v: Vec<i64> = parts.iter().flat_map(|p| if let Value::I(v, _) = p { v.clone() } else { Vec::new() }).collect();
                        let len = v.len();
                        Value::I(v, vec![len])
                    } else {
                        let ts: Vec<Tensor> = parts.into_iter().map(|p| if let Value::F(t) = p { Ok(t) } else { Err("Concat of mixed values".to_string()) }).collect::<Result<_, _>>()?;
                        let r = ts[0].shape.len();
                        let ax = axis(n.int("axis").unwrap_or(0), r)?;
                        let mut shape = ts[0].shape.clone();
                        shape[ax] = ts.iter().map(|t| t.shape[ax]).sum();
                        let y = new(&shape, scratch);
                        let os = strides(&shape);
                        let mut off = 0;
                        for t in &ts {
                            self.copy(gpu, enc, &t.buf, &y.buf, &t.shape, &strides(&t.shape), &os, 0, off * os[ax], &mut keep)?;
                            off += t.shape[ax];
                        }
                        Value::F(y)
                    }
                }
                ("MaxPool" | "AveragePool", _) => Value::F(self.pool(gpu, enc, n, &float(0)?, scratch, &mut keep)?),
                ("Reshape", _) => {
                    let a = float(0)?;
                    let want = ints(1)?;
                    let total: usize = a.shape.iter().product();
                    let mut shape: Vec<usize> = want.iter().enumerate().map(|(d, &s)| if s == 0 { a.shape.get(d).copied().unwrap_or(1) } else { s.max(1) as usize }).collect();
                    if let Some(p) = want.iter().position(|&s| s == -1) {
                        let known: usize = shape.iter().enumerate().filter(|(d, _)| *d != p).map(|(_, s)| s).product();
                        shape[p] = total / known.max(1);
                    }
                    if shape.iter().product::<usize>() != total {
                        return Err(format!("Reshape {:?} to {want:?}", a.shape));
                    }
                    Value::F(Tensor { buf: a.buf, shape })
                }
                ("Squeeze" | "Unsqueeze", _) => {
                    let axes = if let Some(a) = n.ints("axes") { a } else { ints(1)? };
                    match arg(0)? {
                        Value::F(a) => {
                            let shape = if n.op == "Squeeze" {
                                let ax: Vec<usize> = axes.iter().map(|&x| axis(x, a.shape.len())).collect::<Result<_, _>>()?;
                                a.shape.iter().enumerate().filter(|(d, _)| !ax.contains(d)).map(|(_, &s)| s).collect()
                            } else {
                                let r = a.shape.len() + axes.len();
                                let ax: Vec<usize> = axes.iter().map(|&x| axis(x, r)).collect::<Result<_, _>>()?;
                                let mut it = a.shape.iter();
                                (0..r).map(|d| if ax.contains(&d) { 1 } else { *it.next().unwrap_or(&1) }).collect()
                            };
                            Value::F(Tensor { buf: a.buf, shape })
                        }
                        Value::I(v, s) => {
                            let shape = if n.op == "Squeeze" { s.into_iter().filter(|&d| d != 1).collect() } else { vec![v.len()] };
                            Value::I(v, shape)
                        }
                    }
                }
                ("Shape", _) => {
                    let a = arg(0)?;
                    let s = match a {
                        Value::F(t) => t.shape,
                        Value::I(_, s) => s,
                    };
                    let v: Vec<i64> = s.iter().map(|&d| d as i64).collect();
                    let len = v.len();
                    Value::I(v, vec![len])
                }
                ("Gather", _) => {
                    let (data, idx) = (ints(0)?, ints(1)?);
                    let v: Vec<i64> = idx.iter().map(|&i| data.get(if i < 0 { (i + data.len() as i64) as usize } else { i as usize }).copied().unwrap_or(0)).collect();
                    let len = v.len();
                    Value::I(v, if len == 1 { vec![] } else { vec![len] })
                }
                ("Cast", _) => arg(0)?,
                (other, _) => return Err(format!("unsupported op {other}")),
            };
            if std::env::var_os("NDPROFILE").is_some() {
                let done = std::mem::replace(enc, gpu.encoder());
                gpu.submit_wait(done);
                let label = match self.fused.get(&idx) {
                    Some(Fused::Conv(c, _, _)) => format!("Conv {}x{} g{}", c.kh, c.kw, if c.group == 1 { "1".to_string() } else { "dw".into() }),
                    Some(Fused::Linear { n, .. }) => format!("Linear ->{n}"),
                    _ => n.op.clone(),
                };
                let mut prof = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
                let e = prof.entry(label).or_insert((0.0, 0));
                e.0 += node_start.elapsed().as_secs_f64() * 1e3;
                e.1 += 1;
            }
            if std::env::var_os("NDTRACE").is_some() {
                let shape = match &out {
                    Value::F(t) => format!("{:?}", t.shape),
                    Value::I(v, _) => format!("ints {v:?}"),
                };
                eprintln!("{idx:4} {:<18} -> {shape}", n.op);
            }
            values.insert(n.outputs[0].clone(), out);
        }
        // Parameter buffers stay alive in wgpu until the work that uses them is done.
        drop(keep);
        match values.remove(&self.output) {
            Some(Value::F(t)) => Ok(t),
            _ => Err(format!("output {} not computed", self.output)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &Buf, w: &Buf, b: &Buf, y: &Buf, mkib: [usize; 4], strides: [usize; 3], act: u32, keep: &mut Vec<Buf>) {
        let [m, k, i, batch] = mkib;
        let d = gpu.uniform(&[m as u32, k as u32, i as u32, act, strides[0] as u32, strides[1] as u32, strides[2] as u32, 0]);
        gpu.dispatch3(enc, "nd_bgemm", &[x, w, b, y, &d], (m.div_ceil(64) as u32, k.div_ceil(64) as u32, batch as u32));
        keep.push(d);
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &Buf, y: &Buf, dims: &[usize], is: &[usize], os: &[usize], ibase: usize, obase: usize, keep: &mut Vec<Buf>) -> Result<(), String> {
        if dims.len() > RANK {
            return Err(format!("rank {} > {RANK}", dims.len()));
        }
        let total: usize = dims.iter().product();
        let mut p = vec![0u32; 22];
        p[0] = total as u32;
        p[1] = dims.len() as u32;
        p[2] = ibase as u32;
        p[3] = obase as u32;
        for d in 0..dims.len() {
            p[4 + d] = dims[d] as u32;
            p[10 + d] = is[d] as u32;
            p[16 + d] = os[d] as u32;
        }
        let pb = gpu.upload_u32(&p);
        gpu.elementwise(enc, "nd_copy", &[x, y, &pb], total);
        keep.push(pb);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, op: &str, a: &Tensor, b: &Tensor, scratch: &mut Vec<Buf>, keep: &mut Vec<Buf>) -> Result<Tensor, String> {
        let r = a.shape.len().max(b.shape.len());
        if r > RANK {
            return Err(format!("rank {r} > {RANK}"));
        }
        let pad = |s: &[usize]| -> Vec<usize> { std::iter::repeat_n(1, r - s.len()).chain(s.iter().copied()).collect() };
        let (sa, sb) = (pad(&a.shape), pad(&b.shape));
        let shape: Vec<usize> = sa.iter().zip(&sb).map(|(&x, &y)| x.max(y)).collect();
        if sa.iter().zip(&sb).any(|(&x, &y)| x != y && x != 1 && y != 1) {
            return Err(format!("{op}: shapes {:?} and {:?} do not broadcast", a.shape, b.shape));
        }
        let bstr = |s: &[usize]| -> Vec<usize> { strides(s).iter().zip(s).map(|(&st, &d)| if d == 1 { 0 } else { st }).collect() };
        let total: usize = shape.iter().product();
        let mut p = vec![0u32; 22];
        p[0] = total as u32;
        p[1] = r as u32;
        p[2] = match op {
            "Add" => 0,
            "Sub" => 1,
            "Mul" => 2,
            "Div" => 3,
            _ => 4,
        };
        let (astr, bstrv) = (bstr(&sa), bstr(&sb));
        for d in 0..r {
            p[4 + d] = shape[d] as u32;
            p[10 + d] = astr[d] as u32;
            p[16 + d] = bstrv[d] as u32;
        }
        let pb = gpu.upload_u32(&p);
        let y = gpu.storage(total.max(1));
        scratch.push(y.clone());
        gpu.elementwise(enc, "nd_binary", &[&a.buf, &b.buf, &y, &pb], total);
        keep.push(pb);
        Ok(Tensor { buf: y, shape })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, a: &Tensor, c: &ConvW, act: u32, res: Option<&Tensor>, scratch: &mut Vec<Buf>, keep: &mut Vec<Buf>) -> Result<Tensor, String> {
        let [n, cin, h, w] = <[usize; 4]>::try_from(a.shape.clone()).map_err(|_| format!("Conv input {:?}", a.shape))?;
        let [sh, sw] = c.stride;
        // Leading and trailing padding; SAME_UPPER puts the odd one at the end.
        let (pt, pl, pb, pr) = if c.same_upper {
            let (oh, ow) = (h.div_ceil(sh), w.div_ceil(sw));
            let th = ((oh - 1) * sh + c.kh).saturating_sub(h);
            let tw = ((ow - 1) * sw + c.kw).saturating_sub(w);
            (th / 2, tw / 2, th - th / 2, tw - tw / 2)
        } else {
            let p = c.pads.unwrap_or([0; 4]);
            (p[0], p[1], p[2], p[3])
        };
        let oh = (h + pt + pb - c.kh) / sh + 1;
        let ow = (w + pl + pr - c.kw) / sw + 1;
        let y = gpu.storage(n * c.cout * oh * ow);
        scratch.push(y.clone());
        if let Some(r) = res
            && r.shape != [n, c.cout, oh, ow]
        {
            return Err(format!("residual {:?} for a conv giving {:?}", r.shape, [n, c.cout, oh, ow]));
        }
        if res.is_some() && c.group != 1 {
            return Err("residual after a grouped conv".into());
        }
        if c.group == 1 {
            if c.cin_g != cin {
                return Err(format!("Conv expects {} channels, has {cin}", c.cin_g));
            }
            let p = gpu.uniform(&[n, cin, h, w, c.cout, oh, ow, c.kh, c.kw, pt, sh, act as usize, pl, sw, usize::from(res.is_some()), 0].map(|v| v as u32));
            // The 128-pixel tile where it pays (NO128 keeps 64, better on RDNA 3.5).
            let wide = c.fast && std::env::var_os("NO128").is_none();
            let (pipe, mtile) = if wide { ("conv128", 128) } else if c.fast { ("conv_fast", 64) } else { ("conv", 64) };
            let rb = res.map_or(&a.buf, |r| &r.buf);
            gpu.dispatch(enc, pipe, &[&a.buf, &c.w, &c.b, &y, &p, rb], ((n * oh * ow).div_ceil(mtile) as u32, c.cout.div_ceil(64) as u32));
            keep.push(p);
        } else if c.cin_g == 1 && c.group == cin && c.group == c.cout {
            let p = gpu.uniform(&[n, cin, h, w, oh, ow, c.kh, c.kw, pt, sh, act as usize, pl, sw].map(|v| v as u32));
            gpu.elementwise(enc, "depthwise", &[&a.buf, &c.w, &c.b, &y, &p], n * cin * oh * ow);
            keep.push(p);
        } else {
            return Err(format!("grouped conv with group {}", c.group));
        }
        Ok(Tensor { buf: y, shape: vec![n, c.cout, oh, ow] })
    }

    #[allow(clippy::too_many_arguments)]
    fn pool(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, node: &Node, a: &Tensor, scratch: &mut Vec<Buf>, keep: &mut Vec<Buf>) -> Result<Tensor, String> {
        let [n, c, h, w] = <[usize; 4]>::try_from(a.shape.clone()).map_err(|_| format!("pool input {:?}", a.shape))?;
        let k = node.ints("kernel_shape").ok_or("pool without kernel_shape")?;
        let (kh, kw) = (k[0] as usize, k[1] as usize);
        let s = node.ints("strides").unwrap_or_else(|| vec![1, 1]);
        let (sh, sw) = (s[0] as usize, s[1] as usize);
        if node.int("ceil_mode").unwrap_or(0) != 0 {
            return Err("pool with ceil_mode".into());
        }
        if node.op == "AveragePool" && node.int("count_include_pad").unwrap_or(0) != 0 {
            return Err("AveragePool counting padding".into());
        }
        let (pt, pl, pb, pr) = if node.string("auto_pad").as_deref() == Some("SAME_UPPER") {
            let (oh, ow) = (h.div_ceil(sh), w.div_ceil(sw));
            let th = ((oh - 1) * sh + kh).saturating_sub(h);
            let tw = ((ow - 1) * sw + kw).saturating_sub(w);
            (th / 2, tw / 2, th - th / 2, tw - tw / 2)
        } else {
            let p = node.ints("pads").unwrap_or_else(|| vec![0; 4]);
            (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize)
        };
        let oh = (h + pt + pb - kh) / sh + 1;
        let ow = (w + pl + pr - kw) / sw + 1;
        let y = gpu.storage(n * c * oh * ow);
        scratch.push(y.clone());
        let avg = usize::from(node.op == "AveragePool");
        let p = gpu.uniform(&[n * c, h, w, oh, ow, kh, kw, sh, sw, pt, pl, avg].map(|v| v as u32));
        gpu.elementwise(enc, "nd_pool", &[&a.buf, &y, &p], n * c * oh * ow);
        keep.push(p);
        Ok(Tensor { buf: y, shape: vec![n, c, oh, ow] })
    }

    /// The arg-max over the last axis of `logits`, per row, read back.
    pub fn argmax(&self, gpu: &Gpu, mut enc: wgpu::CommandEncoder, logits: &Tensor) -> Vec<u32> {
        let len = *logits.shape.last().unwrap_or(&1);
        let rows = logits.shape.iter().product::<usize>() / len.max(1);
        let y = gpu.storage(rows);
        let p = gpu.uniform(&[rows as u32, len as u32, 0, 0]);
        let groups = (rows.min(65535) as u32, rows.div_ceil(65535) as u32);
        gpu.dispatch(&mut enc, "nd_argmax", &[&logits.buf, &y, &p], groups);
        let out = gpu.read(enc, &y, rows);
        gpu.recycle([y]);
        out.into_iter().map(f32::to_bits).collect()
    }
}
