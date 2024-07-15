use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct EqualLayer;
impl Layer for EqualLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    assert!(*constants[1].unwrap().first().unwrap() == Fr::from(0));
    let mut graph = Graph::new();
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(1); *input_shapes[0].last().unwrap()]).into_dyn(),
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let one_output = graph.addNode(one, vec![]);
    let sub_output = graph.addNode(sub, vec![(one_output, 0), (-1, 0)]);
    graph.outputs.push((sub_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)])
  }
}
