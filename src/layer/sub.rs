use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;

pub struct SubLayer;
impl Layer for SubLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add_output = graph.addNode(add, vec![(-1, 0), (-2, 0)]);
    graph.outputs.push((add_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 1)])
  }
}
