use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{Array1, ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct DivLayer;
impl Layer for DivLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    if let Some(c) = constants[1] {
      let c = onnx::SF_FLOAT * util::fr_to_int(*c.first().unwrap()) as f32;
      let div = graph.addBB(Box::new(DivConstBasicBlock { c: c }));
      let div_check = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQ2BasicBlock {
          setup: Some((Box::new(DivConstBasicBlock { c: c }), onnx::CQ_RANGE_LOWER, onnx::CQ_RANGE)),
        }),
        N: 1,
      }));
      let div_output = graph.addNode(div, vec![(-1, 0)]);
      let _ = graph.addNode(div_check, vec![(-1, 0), (div_output, 0)]);
      graph.outputs.push((div_output, 0));
      return (graph, vec![input_shapes[0].clone()]);
    }

    assert!(*input_shapes[1].last().unwrap() == 1);

    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock { output_SF: onnx::SF }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: Array1::from_iter(0..1 << 9).map(|x| Fr::from(*x)),
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: onnx::SF * 2 }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));
    // a is dividend (first input)
    // b is divisor (second input)
    // q and r are quotient and remainder for (2*a*SF + b) / (2 * b)
    // We first check that a*SF*2+b = q*b*2+r:
    let div_output = graph.addNode(div, vec![(-1, 0), (-2, 0)]);
    let a_SF2 = graph.addNode(mul_SF2, vec![(-1, 0)]);
    let a_SF2_plus_b = graph.addNode(add, vec![(a_SF2, 0), (-2, 0)]);
    let b2 = graph.addNode(mul_2, vec![(-2, 0)]);
    let qb2 = graph.addNode(mul, vec![(div_output, 0), (b2, 0)]);
    let qb2_plus_r = graph.addNode(add, vec![(qb2, 0), (div_output, 1)]);
    let _ = graph.addNode(eq, vec![(a_SF2_plus_b, 0), (qb2_plus_r, 0)]);
    // Now check r≥0:
    let _ = graph.addNode(range_check, vec![(div_output, 1)]);
    // Now check 2b-r≥0:
    let b2_minus_r = graph.addNode(sub, vec![(b2, 0), (div_output, 1)]);
    let _ = graph.addNode(range_check, vec![(b2_minus_r, 0)]);
    graph.outputs.push((div_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
