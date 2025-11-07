use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bls12_381::Fr;
use ark_std::Zero;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct NonzeroLayer;

impl Layer for NonzeroLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_elem_num = input_shapes[0].iter().product::<usize>();
    let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::zero(),
        shape: vec![1, input_elem_num],
      }));
    let output = graph.addNode(constantOfShape, vec![]);
    graph.outputs.push((output, 0));
    (graph, vec![vec![1, input_elem_num]], vec![input_types[0]])
  }
}
