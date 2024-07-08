use crate::graph::Graph;
pub use arithmetic::{AddLayer, SubLayer};
use ark_bn254::Fr;
pub use cast::CastLayer;
pub use concat::ConcatLayer;
pub use div::DivLayer;
pub use equal::EqualLayer;
pub use expand::ExpandLayer;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
pub use mul::MulLayer;
use ndarray::ArrayD;
pub use nonlinear::{ErfLayer, ReLULayer, SqrtLayer};
pub use pow::PowLayer;
pub use r#where::WhereLayer;
pub use reducemean::ReduceMeanLayer;
pub use reshape::ReshapeLayer;
pub use shape::ShapeLayer;
pub use softmax::SoftmaxLayer;
pub use squeeze::{SqueezeLayer, UnsqueezeLayer};
use tract_onnx::pb::AttributeProto;
pub use transpose::TransposeLayer;

pub mod arithmetic;
pub mod cast;
pub mod concat;
pub mod div;
pub mod equal;
pub mod expand;
pub mod gather;
pub mod matmul;
pub mod mul;
pub mod nonlinear;
pub mod pow;
pub mod reducemean;
pub mod reshape;
pub mod shape;
pub mod softmax;
pub mod squeeze;
pub mod transpose;
pub mod r#where;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>);
}
