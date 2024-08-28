use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

#[derive(Debug)]
pub struct ShapeBasicBlock;
impl BasicBlock for ShapeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let shape: Vec<_> = inputs[0].shape().iter().map(|&x| Fr::from(x as i32)).collect();
    let shape = arr1(&shape).into_dyn();
    let padded_shape = util::pad_to_pow_of_two(&shape, &Fr::zero());
    vec![padded_shape]
  }
}

pub struct ShapeLayer;
impl Layer for ShapeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let shape = graph.addBB(Box::new(ShapeBasicBlock {}));
    let shape_output = graph.addNode(shape, vec![(-1, 0)]);
    graph.outputs.push((shape_output, 0));
    (graph, vec![vec![input_shapes[0].len()]], vec![DatumType::I64])
  }
}
