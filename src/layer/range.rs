use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{Array1, ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct RangeLayer;
impl Layer for RangeLayer {
  fn graph(_input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let mut all_is_constant = true;

    let start = util::fr_to_int(constants[0].unwrap().as_slice().unwrap()[0]);
    // constants[1] (limit) might be None or Some
    let limit = if constants[1].is_none() {
      all_is_constant = false;
      start
    } else {
      util::fr_to_int(constants[1].unwrap().as_slice().unwrap()[0])
    };
    let delta = util::fr_to_int(constants[2].unwrap().as_slice().unwrap()[0]);
    // currently only support positive delta
    assert!(delta >= 0);

    if all_is_constant {
      // all fields are constant, no need to prove anything
      let range = graph.addBB(Box::new(RangeConstBasicBlock {
        start: start,
        limit: limit,
        delta: delta,
      }));
      let range_output = graph.addNode(range, vec![]);
      graph.outputs.push((range_output, 0));
    } else {
      // currently only consider the case where limit is not a constant
      let range = graph.addBB(Box::new(RangeBasicBlock { start: start, delta: delta }));
      let nonnegative_check = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQBasicBlock {
          setup: Array1::from_iter(0..onnx::CQ_RANGE).map(|x| Fr::from(*x as i32)),
        }),
        N: 1,
      }));
      let sub = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(SubBasicBlock {}),
        N: 1,
      }));
      let range_output = graph.addNode(range, vec![(-1, 0)]);
      let sub_output = graph.addNode(sub, vec![(-1, 0), (range_output, 0)]); // limit - range_output
      let _ = graph.addNode(nonnegative_check, vec![(sub_output, 0)]); // check if the diff is not negative
      graph.outputs.push((range_output, 0));
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
