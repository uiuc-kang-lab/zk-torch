use crate::graph::Graph;
pub use add::AddLayer;
pub use gather::GatherLayer;
pub use matmul::MatMulLayer;
pub use reducemean::ReduceMeanLayer;
pub use relu::ReLULayer;
pub use sub::SubLayer;

pub mod add;
pub mod gather;
pub mod matmul;
pub mod reducemean;
pub mod relu;
pub mod sub;

pub trait Layer {
  fn graph(input_shapes: &Vec<&Vec<usize>>) -> (Graph, Vec<Vec<usize>>);
}
