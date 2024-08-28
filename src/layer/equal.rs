use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct EqualLayer;
impl Layer for EqualLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(1); util::next_pow(*input_shapes[0].last().unwrap() as u32) as usize]).into_dyn(),
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let equal = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(ElementwiseEqBasicBlock {}),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));

    let r: Vec<_> = (*onnx::CQ_RANGE_LOWER..-*onnx::CQ_RANGE_LOWER + 1).filter(|&x| x != 0).map(Fr::from).collect();
    let nonzero_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { setup: arr1(&r) }),
      N: 1,
    }));

    let equal_output = graph.addNode(equal, vec![(-1, 0), (-2, 0)]); // a == b
    let one_output = graph.addNode(one, vec![]);
    let not_equal_output = graph.addNode(sub, vec![(one_output, 0), (equal_output, 0)]);

    let a_equal_b = graph.addNode(mul, vec![(-1, 0), (equal_output, 0)]); // a * (a == b)
    let b_equal_a = graph.addNode(mul, vec![(-2, 0), (equal_output, 0)]); // b * (a == b)
    let a_minus_b = graph.addNode(sub, vec![(-1, 0), (-2, 0)]); // a - b
    let a_not_equal_b = graph.addNode(mul, vec![(a_minus_b, 0), (not_equal_output, 0)]); // (a - b) * (1 - (a == b))
    let add_output = graph.addNode(add, vec![(a_not_equal_b, 0), (equal_output, 0)]); // should be all nonzeros

    let _eq_check = graph.addNode(eq, vec![(a_equal_b, 0), (b_equal_a, 0)]);
    let _nonzero_check = graph.addNode(nonzero_check, vec![(add_output, 0)]);

    graph.outputs.push((equal_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}
