use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

#[derive(Debug)]
pub struct ExpandBasicBlock;
impl BasicBlock for ExpandBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let newShape: Vec<_> = inputs[1].as_slice().unwrap().iter().map(|&x| util::fr_to_int(x) as usize).collect();
    vec![inputs[0].broadcast(newShape).unwrap().into_owned()]
  }
}

pub struct ExpandLayer;
impl Layer for ExpandLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    assert!(Fr::from(*input_shapes[0].last().unwrap() as i32) == *constants[1].unwrap().last().unwrap());
    let mut graph = Graph::new();
    let expand = graph.addBB(Box::new(ExpandBasicBlock {}));
    let expand_output = graph.addNode(expand, vec![(-1, 0), (-2, 0)]);
    graph.outputs.push((expand_output, 0));
    let newShape: Vec<_> = constants[1].unwrap().as_slice().unwrap().iter().map(|&x| util::fr_to_int(x) as usize).collect();
    (graph, vec![newShape])
  }
}
