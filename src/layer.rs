use crate::graph::Graph;
pub use add::AddLayer;
use ark_bn254::Fr;
pub use cast::CastLayer;
pub use div::DivLayer;
pub use equal::EqualLayer;
pub use erf::ErfLayer;
pub use expand::ExpandLayer;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
pub use mul::MulLayer;
use ndarray::ArrayD;
pub use pow::PowLayer;
pub use r#where::WhereLayer;
pub use reducemean::ReduceMeanLayer;
pub use relu::ReLULayer;
pub use reshape::ReshapeLayer;
pub use shape::ShapeLayer;
pub use softmax::SoftmaxLayer;
pub use sqrt::SqrtLayer;
pub use sub::SubLayer;
use tract_onnx::pb::AttributeProto;
pub use transpose::TransposeLayer;

pub mod add;
pub mod cast;
pub mod div;
pub mod equal;
pub mod erf;
pub mod expand;
pub mod gather;
pub mod matmul;
pub mod mul;
pub mod pow;
pub mod reducemean;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod softmax;
pub mod sqrt;
pub mod sub;
pub mod transpose;
pub mod r#where;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>);
}
