use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct RangeLayer;
impl Layer for RangeLayer {
  fn graph(_input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let start = constants[0].unwrap().as_slice().unwrap()[0];
    let limit = constants[1].unwrap().as_slice().unwrap()[0];
    let delta = constants[2].unwrap().as_slice().unwrap()[0];
    // we may need to prove this
    let range = graph.addBB(Box::new(RangeBasicBlock {
      start: start,
      limit: limit,
      delta: delta,
    }));
    let range_check = graph.addBB(Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(RangeBasicBlock {
            start: start,
            limit: limit,
            delta: delta,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }));
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
