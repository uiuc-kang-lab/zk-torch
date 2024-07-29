use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct NegLayer;

impl Layer for NegLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let zero = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::zero(); *input_shapes[0].last().unwrap()]).into_dyn(),
    }));
    let layer = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let zero_output = graph.addNode(zero, vec![]);
    let layer_output = graph.addNode(layer, vec![(zero_output, 0), (-1, 0)]);
    graph.outputs.push((layer_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)])
  }
}
