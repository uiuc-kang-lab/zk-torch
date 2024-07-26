use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::Array1;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;

pub struct RangeLayer;
impl Layer for RangeLayer {
  fn graph(_input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let start = constants[0].unwrap().as_slice().unwrap()[0];
    let limit = constants[1].unwrap().as_slice().unwrap()[0];
    let delta = constants[2].unwrap().as_slice().unwrap()[0];

    let range = graph.addBB(Box::new(RangeBasicBlock {
      start: start,
      limit: limit,
      delta: delta,
    }));
    let (empty, empty1) = (ArrayD::zeros(IxDyn(&[0])), ArrayD::zeros(IxDyn(&[0])));
    let empty_input = vec![&empty1];
    let range_tensor = &RangeBasicBlock {
      start: start,
      limit: limit,
      delta: delta,
    }
    .run(&empty, &empty_input)[0]
      .clone();

    let range_setup: Array1<Fr> = util::pad_to_pow_of_two(&range_tensor, &Fr::zero()).into_dimensionality::<ndarray::Ix1>().unwrap();
    let range_check = graph.addBB(Box::new(CQBasicBlock { setup: range_setup }));
    let range_output = graph.addNode(range, vec![(-1, 0)]);
    let _ = graph.addNode(range_check, vec![(-1, 0), (range_output, 0)]);
    graph.outputs.push((range_output, 0));

    let mut length = 0;
    let mut start = start;
    while start < limit {
      start += delta;
      length += 1;
    }

    (graph, vec![vec![length]])
  }
}
