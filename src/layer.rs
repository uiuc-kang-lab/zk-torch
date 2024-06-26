use crate::graph::Graph;
use ark_bn254::Fr;
pub use mul::MulLayer;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub mod mul;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>);
}
