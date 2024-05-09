use crate::basic_block::*;
use crate::graph::*;

pub fn graph() -> Graph {
  Graph {
    basic_blocks: vec![Box::new(AddBasicBlock {})],
    nodes: vec![Node {
      basic_block: 0,
      inputs: vec![(-1, 0), (-2, 0)],
    }],
    outputs: vec![(0, 0)],
  }
}
