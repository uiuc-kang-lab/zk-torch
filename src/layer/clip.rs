use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;

use super::equal;

pub struct ClipLayer;
impl Layer for ClipLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let min = constants[1].unwrap().as_slice().unwrap()[0];
    let max = constants[2].unwrap().as_slice().unwrap()[0];

    let clip = graph.addBB(Box::new(ClipBasicBlock {
      min: min,
      max: max,
    }));
    let clip_output = graph.addNode(clip, vec![(-1, 0)]);
    // todo: clip check, it should be ready after MaxBasicBlock is proved

    graph.outputs.push((clip_output, 0));

    (graph, vec![input_shapes[0].clone()])
  }
}
