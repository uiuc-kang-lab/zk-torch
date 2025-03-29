use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::One;
use ndarray::{arr1, ArrayD};
use rayon::iter::ParallelIterator;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct SqrtLayer;
impl Layer for SqrtLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: sf_log,
      output_SF: sf_log,
    }));
    let sf = onnx::SF.read().unwrap().to_owned();
    let sf_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(sf as i32)]).into_dyn(),
    }));
    let two_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(2)]).into_dyn(),
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let len = util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize;
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock { len }),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let non_negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
        setup: util::CQArrayType::NonPositive,
      }),
      N: 1,
    }));

    // SqrtBB(x) = sqrt(x/SF)*SF + eps (where -1 < eps < 1)
    let sqrt_output = graph.addNode(sqrt, vec![(-1, 0)]);
    // The following operations are to check if sqrt_output is correct
    // square_sqrt = SqrtBB(x)^2 = x*SF + 2*sqrt(x/SF)*SF*eps + eps^2
    let square_sqrt = graph.addNode(mul, vec![(sqrt_output, 0), (sqrt_output, 0)]);
    // scale_input_by_sf = x*SF
    let sf_const_output = graph.addNode(sf_const, vec![]);
    let scale_input_by_sf = graph.addNode(mul_scalar, vec![(-1, 0), (sf_const_output, 0)]);
    // difference = SqrtBB(x)^2 - x*SF = 2*sqrt(x/SF)*SF*eps + eps^2 = 2*SqrtBB(x)*eps - eps^2
    // Because -1 < eps < 1, -2*SqrtBB(x) < 2*SqrtBB(x)*eps < 2*SqrtBB(x) and -1 < -eps^2 < 0.
    // Therefore, -1 - 2*SqrtBB(x) < difference < 2*SqrtBB(x).
    // The following two inequalities should hold:
    // 1. difference + 2*SqrtBB(x) >= 0
    // 2. difference - 2*SqrtBB(x) < 0
    let difference = graph.addNode(sub, vec![(square_sqrt, 0), (scale_input_by_sf, 0)]);
    // scale_output_by_2 = 2*SqrtBB(x)
    let two_const_output = graph.addNode(two_const, vec![]);
    let scale_output_by_2 = graph.addNode(mul_scalar, vec![(sqrt_output, 0), (two_const_output, 0)]);
    // d_plus_scale_output_by_2 = difference + 2*SqrtBB(x)
    let d_plus_scale_output_by_2 = graph.addNode(add, vec![(difference, 0), (scale_output_by_2, 0)]);
    // d_minus_scale_output_by_2 = difference - 2*SqrtBB(x)
    let d_minus_scale_output_by_2 = graph.addNode(sub, vec![(difference, 0), (scale_output_by_2, 0)]);
    let _ = graph.addNode(non_negative_check, vec![(d_plus_scale_output_by_2, 0)]);
    let _ = graph.addNode(negative_check, vec![(d_minus_scale_output_by_2, 0)]);

    graph.outputs.push((sqrt_output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
