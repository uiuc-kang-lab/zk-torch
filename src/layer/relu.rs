use crate::basic_block::*;
use crate::graph::*;
use std::collections::HashMap;

pub fn graph() -> Graph {
  Graph {
    basic_blocks: vec![
      Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }),
      Box::new(CQ2BasicBlock {
        table_dict: HashMap::new(),
        setup: Some((Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }), -(1 << 5), 1 << 6)),
      }),
    ],
    nodes: vec![
      Node {
        basic_block: 0,
        inputs: vec![(-1, 0)],
      },
      Node {
        basic_block: 1,
        inputs: vec![(-1, 0), (0, 0)],
      },
    ],
    outputs: vec![(0, 0)],
  }
}
