//! wgpu-ocr: the [ocrs](https://github.com/robertknight/ocrs) text detection and recognition
//! models on wgpu compute shaders (WGSL), with the graph and weights read straight from the
//! `.onnx` files. A prototype: see the README for what it does, how it was measured, and what it
//! does not do yet.
//!
//! - [`onnx`]: the model graph and weights, read with `rten-onnx`.
//! - [`gpu`]: the device, the kernels, a graph executor for convolutional graphs, the detector
//!   and the recogniser (convolutional trunk, bidirectional GRU, output layer).

pub mod gpu;
pub mod onnx;
