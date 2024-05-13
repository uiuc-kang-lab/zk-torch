use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use std::collections::HashMap;

pub struct MatMulLayer;
impl Layer for MatMulLayer {
  fn graph() -> Graph {
    let mut graph = Graph::new();
    let matmul = graph.addBB(Box::new(MatMulBasicBlock {}));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock { input_SF: 6, output_SF: 3 }));
    let change_SF_check = graph.addBB(Box::new(CQ2BasicBlock {
      table_dict: HashMap::new(),
      setup: Some((Box::new(ChangeSFBasicBlock { input_SF: 6, output_SF: 3 }), -(1 << 5), 1 << 6)),
    }));
    let matmul_output = graph.addNode(matmul, vec![(-1, 0), (-2, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(matmul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (change_SF_output, 0)]);
    graph.outputs.push((change_SF_output, 0));
    graph
  }
}
