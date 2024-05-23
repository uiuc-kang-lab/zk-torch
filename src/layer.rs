use crate::graph::Graph;
pub use add::AddLayer;
use ark_bn254::Fr;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
use ndarray::ArrayD;
pub use pow::PowLayer;
pub use reducemean::ReduceMeanLayer;
pub use relu::ReLULayer;
pub use sub::SubLayer;
pub use sqrt::SqrtLayer;

pub mod add;
pub mod gather;
pub mod matmul;
pub mod pow;
pub mod sqrt;
pub mod reducemean;
pub mod relu;
pub mod sub;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>) -> (Graph, Vec<Vec<usize>>);
}
