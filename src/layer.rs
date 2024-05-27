use crate::graph::Graph;
pub use add::AddLayer;
use ark_bn254::Fr;
pub use div::DivLayer;
pub use expand::ExpandLayer;
pub use cast::CastLayer;
pub use erf::ErfLayer;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
pub use mul::MulLayer;
use ndarray::ArrayD;
pub use pow::PowLayer;
pub use reducemean::ReduceMeanLayer;
pub use relu::ReLULayer;
pub use reshape::ReshapeLayer;
pub use shape::ShapeLayer;
pub use sqrt::SqrtLayer;
pub use sub::SubLayer;
pub use equal::EqualLayer;
pub use r#where::WhereLayer;
use tract_onnx::pb::AttributeProto;
pub use transpose::TransposeLayer;
pub use softmax::SoftmaxLayer;

pub mod add;
pub mod erf;
pub mod softmax;
pub mod r#where;
pub mod cast;
pub mod expand;
pub mod equal;
pub mod div;
pub mod gather;
pub mod matmul;
pub mod mul;
pub mod pow;
pub mod reducemean;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod sqrt;
pub mod sub;
pub mod transpose;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>);
}
