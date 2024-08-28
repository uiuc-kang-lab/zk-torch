use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct WhereLayer;
impl Layer for WhereLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    //condition, X, Y
    //condition * X + (1-condition) * Y
    let mut graph = Graph::new();
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(1); util::next_pow(*input_shapes[0].last().unwrap() as u32) as usize]).into_dyn(),
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));

    let one_output = graph.addNode(one, vec![]);
    let mul1_output = graph.addNode(if input_shapes[1].len() == 0 { mul_scalar } else { mul }, vec![(-1, 0), (-2, 0)]);
    let sub_output = graph.addNode(sub, vec![(one_output, 0), (-1, 0)]);
    let mul2_output = graph.addNode(if input_shapes[2].len() == 0 { mul_scalar } else { mul }, vec![(sub_output, 0), (-3, 0)]);
    let add_output = graph.addNode(add, vec![(mul1_output, 0), (mul2_output, 0)]);
    graph.outputs.push((add_output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
