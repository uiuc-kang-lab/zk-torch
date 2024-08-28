use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::One;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct NotLayer;

impl Layer for NotLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::one(); *input_shapes[0].last().unwrap()]).into_dyn(),
    }));
    let layer = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let _ = graph.addNode(bool_check, vec![(-1, 0)]);
    let one_output = graph.addNode(one, vec![]);
    let layer_output = graph.addNode(layer, vec![(one_output, 0), (-1, 0)]);
    graph.outputs.push((layer_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}
