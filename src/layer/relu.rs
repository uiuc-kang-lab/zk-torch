use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use std::collections::HashMap;

pub struct ReLULayer;
impl Layer for ReLULayer {
  fn graph() -> Graph {
    let mut graph = Graph::new();
    let relu = graph.addBB(Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }));
    let relu_check = graph.addBB(Box::new(CQ2BasicBlock {
      table_dict: HashMap::new(),
      setup: Some((Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }), -(1 << 5), 1 << 6)),
    }));
    let relu_output = graph.addNode(relu, vec![(-1, 0)]);
    let _ = graph.addNode(relu_check, vec![(-1, 0), (relu_output, 0)]);
    graph.outputs.push((relu_output, 0));
    graph
  }
}
