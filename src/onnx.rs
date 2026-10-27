//! The model graph and weights, read in Rust straight from the `.onnx` file with `rten-onnx`
//! (rten's own parser, already in the dependency tree through `rten`'s `onnx_format`).

use std::collections::HashMap;

use rten_onnx::onnx::{AttributeProto, ModelProto, TensorProto};

/// A float tensor: its dimensions and row-major data.
#[derive(Clone, Debug)]
pub struct Weight {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    attrs: Vec<AttributeProto>,
}

impl Node {
    fn attr(&self, name: &str) -> Option<&AttributeProto> {
        self.attrs.iter().find(|a| a.name.as_deref() == Some(name))
    }

    pub fn ints(&self, name: &str) -> Option<Vec<i64>> {
        self.attr(name).map(|a| a.ints.clone())
    }

    pub fn int(&self, name: &str) -> Option<i64> {
        self.attr(name).and_then(|a| a.i)
    }

    pub fn string(&self, name: &str) -> Option<String> {
        self.attr(name).and_then(|a| a.s.clone())
    }
}

pub struct Onnx {
    pub nodes: Vec<Node>,
    pub weights: HashMap<String, Weight>,
    /// Integer constants (Slice starts/ends/axes and the like).
    pub ints: HashMap<String, Vec<i64>>,
    pub input: String,
    pub output: String,
}

const FLOAT: i32 = 1;
const INT64: i32 = 7;

/// A tensor's values as floats or as integers, from `raw_data` or the typed fields.
fn values(t: &TensorProto) -> Result<(Vec<usize>, Option<Vec<f32>>, Option<Vec<i64>>), String> {
    let dims: Vec<usize> = t.dims.iter().map(|&d| usize::try_from(d).map_err(|_| "negative dimension".to_string())).collect::<Result<_, _>>()?;
    let count: usize = dims.iter().product();
    let raw = t.raw_data.as_ref().map(|r| r.borrow().clone());
    match t.data_type.map(|d| d.0) {
        Some(FLOAT) => {
            let data = match raw {
                Some(bytes) => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                None => t.float_data.clone(),
            };
            if data.len() != count {
                return Err(format!("{:?}: {} values for {count}", t.name, data.len()));
            }
            Ok((dims, Some(data), None))
        }
        Some(INT64) => {
            let data = match raw {
                Some(bytes) => bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect(),
                None => t.int64_data.clone(),
            };
            Ok((dims, None, Some(data)))
        }
        other => Ok((dims, None, None)).map(|v| {
            let _ = other;
            v
        }),
    }
}

impl Onnx {
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let model = ModelProto::parse_buf(&bytes).map_err(|e| format!("{path}: {e:?}"))?;
        let graph = model.graph.ok_or("model has no graph")?;
        let (mut weights, mut ints) = (HashMap::new(), HashMap::new());
        let mut add = |name: String, t: &TensorProto| -> Result<(), String> {
            match values(t)? {
                (dims, Some(data), _) => {
                    weights.insert(name, Weight { dims, data });
                }
                (_, _, Some(i)) => {
                    ints.insert(name, i);
                }
                _ => {}
            }
            Ok(())
        };
        for t in &graph.initializer {
            add(t.name.clone().unwrap_or_default(), t)?;
        }
        let mut nodes = Vec::new();
        for n in &graph.node {
            let op = n.op_type.clone().unwrap_or_default();
            // Constants become weights or integer lists, like initializers.
            if op == "Constant" {
                if let (Some(out), Some(t)) = (n.output.first(), n.attribute.iter().find_map(|a| a.t.as_ref())) {
                    add(out.clone(), t)?;
                }
                continue;
            }
            nodes.push(Node { op, inputs: n.input.clone(), outputs: n.output.clone(), attrs: n.attribute.clone() });
        }
        let input = graph.input.iter().filter_map(|i| i.name.clone()).find(|n| !weights.contains_key(n)).ok_or("no graph input")?;
        let output = graph.output.first().and_then(|o| o.name.clone()).ok_or("no graph output")?;
        Ok(Self { nodes, weights, ints, input, output })
    }

    pub fn weight(&self, name: &str) -> Result<&Weight, String> {
        self.weights.get(name).ok_or_else(|| format!("missing weight {name}"))
    }

    /// How many nodes read `name`.
    pub fn readers(&self, name: &str) -> usize {
        self.nodes.iter().filter(|n| n.inputs.iter().any(|i| i == name)).count()
    }
}
