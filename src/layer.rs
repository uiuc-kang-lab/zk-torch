use crate::graph::Graph;

pub mod add;
pub mod matmul;
pub mod relu;

pub trait Layer {
  fn graph() -> Graph;
}
