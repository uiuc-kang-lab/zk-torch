use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;

pub struct AddLayer;
impl Layer for AddLayer {
  fn graph() -> Graph {
    let mut graph = Graph::new();
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let add_output = graph.addNode(add, vec![(-1, 0), (-2, 0)]);
    graph.outputs.push((add_output, 0));
    graph
  }
}
