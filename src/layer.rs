use crate::graph::Graph;
pub use add::AddLayer;
use ark_bn254::Fr;
pub use div::DivLayer;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
use ndarray::ArrayD;
pub use pow::PowLayer;
pub use reducemean::ReduceMeanLayer;
pub use relu::ReLULayer;
pub use reshape::ReshapeLayer;
pub use sqrt::SqrtLayer;
pub use sub::SubLayer;

pub mod add;
pub mod div;
pub mod gather;
pub mod matmul;
pub mod pow;
pub mod reducemean;
pub mod relu;
pub mod reshape;
pub mod sqrt;
pub mod sub;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>) -> (Graph, Vec<Vec<usize>>);
}
