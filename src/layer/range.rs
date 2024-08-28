use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct RangeLayer;
impl Layer for RangeLayer {
  fn graph(
    _input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let mut all_are_constant = true;

    // check if constants are all Some
    for i in 0..3 {
      if constants[i].is_none() {
        all_are_constant = false;
        break;
      }
    }

    let mut length = 0;
    if all_are_constant {
      let start = util::fr_to_int(constants[0].unwrap().0.as_slice().unwrap()[0]);
      let limit = util::fr_to_int(constants[1].unwrap().0.as_slice().unwrap()[0]);
      let delta = util::fr_to_int(constants[2].unwrap().0.as_slice().unwrap()[0]);

      // all fields are constant
      let range = graph.addBB(Box::new(RangeConstBasicBlock {
        start: start,
        limit: limit,
        delta: delta,
      }));
      let range_output = graph.addNode(range, vec![]);
      graph.outputs.push((range_output, 0));

      let mut start = start;
      while start < limit {
        start += delta;
        length += 1;
      }
    } else {
      panic!("Don't support non-constant range yet");
    }

    (graph, vec![vec![length]], vec![constants[0].unwrap().1])
  }
}
