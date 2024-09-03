use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, Axis};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

#[derive(Debug)]
pub struct GatherBasicBlock;
impl BasicBlock for GatherBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let mut v = Vec::new();
    inputs[1].for_each(|x| {
      let idx = util::fr_to_int(*x) as usize;
      v.extend_from_slice(inputs[0].index_axis(Axis(0), idx).to_slice().unwrap());
    });
    let mut shape = inputs[1].shape().to_vec();
    shape.extend_from_slice(&inputs[0].shape()[1..]);
    let v = ArrayD::from_shape_vec(shape, v).unwrap();
    vec![v]
  }
}

pub struct GatherLayer;
impl Layer for GatherLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let indices = constants[1].unwrap().0.mapv(|x| {
      if x > Fr::from(input_shapes[0][0] as i64) {
        Fr::from(input_shapes[0][0] as i64) + x
      } else {
        x
      }
    });
    let indices = graph.addBB(Box::new(Const2BasicBlock { c: indices }));
    let indices_output = graph.addNode(indices, vec![]);
    let gather = graph.addBB(Box::new(GatherBasicBlock {}));
    let output = graph.addNode(gather, vec![(-1, 0), (indices_output, 0)]);
    graph.outputs.push((output, 0));
    let mut output_shape = input_shapes[1].clone();
    output_shape.extend_from_slice(&input_shapes[0][1..]);
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
