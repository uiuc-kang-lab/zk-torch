use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct AndLayer;
impl Layer for AndLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));

    let _ = graph.addNode(bool_check, vec![(-1, 0)]);
    let _ = graph.addNode(bool_check, vec![(-2, 0)]);
    // If any of the inputs are scalars, use the scalar version of the mul basic block.
    let mul_basicblock = if input_shapes[1].len() == 0 || input_shapes[0].len() == 0 {
      mul_scalar
    // else use the normal version of the mul basic block.
    } else {
      mul
    };
    // If the first input is a scalar, swap the inputs, because the mul scalar basic block expects the scalar to be the second input.
    let and_output = if input_shapes[0].len() == 0 {
      graph.addNode(mul_basicblock, vec![(-2, 0), (-1, 0)])
    } else {
      graph.addNode(mul_basicblock, vec![(-1, 0), (-2, 0)])
    };

    graph.outputs.push((and_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}
