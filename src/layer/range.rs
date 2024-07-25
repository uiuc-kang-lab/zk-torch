use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct RangeLayer;
impl Layer for RangeLayer {
  fn graph(_input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let mut all_are_constant = true;

    let start = util::fr_to_int(constants[0].unwrap().as_slice().unwrap()[0]);
    // constants[1] (limit) might be None or Some
    let limit = if constants[1].is_none() {
      all_are_constant = false;
      start
    } else {
      util::fr_to_int(constants[1].unwrap().as_slice().unwrap()[0])
    };
    let delta = util::fr_to_int(constants[2].unwrap().as_slice().unwrap()[0]);

    if all_are_constant {
      // all fields are constant
      let range = graph.addBB(Box::new(RangeConstBasicBlock {
        start: start,
        limit: limit,
        delta: delta,
      }));
      let range_output = graph.addNode(range, vec![]);
      graph.outputs.push((range_output, 0));
    } else {
      panic!("Don't support non-constant range yet");
    }

    let mut length = 0;
    let mut start = start;
    while start < limit {
      start += delta;
      length += 1;
    }

    (graph, vec![vec![length]])
  }
}
