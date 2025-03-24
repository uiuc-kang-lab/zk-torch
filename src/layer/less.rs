use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Less layer performs `less`, an element-wise logical comparison of two tensors.
pub struct LessLayer;
impl Layer for LessLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    // Inputs: A, B
    // Outputs: L = (A < B); then 1 - L = (A >= B). We can view them as selection of indices.
    // Check 1: (A - B) * L + (-1) * (1 - L) < 0 because A - B will always < 0 at indices of A < B and we set values at other indices as -1
    // Check 1 is equivalent to (A - B) * L - (1 - L) < 0
    // Check 2: 0 * L + (A - B) * (1 - L) >= 0 because A - B will always >= 0 at indices of A >= B and we set values at other indices as 0
    // Check 2 is equivalent to (A - B) * (1 - L) >= 0
    let mut graph = Graph::new();

    let len = util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize;
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock { len }),
      N: 1,
    }));
    let less = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(LessBasicBlock {}),
      N: 1,
    }));
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(1); *input_shapes[0].last().unwrap()]).into_dyn(),
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
        setup: util::CQArrayType::Negative,
      }),
      N: 1,
    }));
    let non_negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    let one_output = graph.addNode(one, vec![]);
    let less_output = graph.addNode(less, vec![(-1, 0), (-2, 0)]);
    let one_minus_less_output = graph.addNode(sub, vec![(one_output, 0), (less_output, 0)]);
    let sub_output = graph.addNode(sub, vec![(-1, 0), (-2, 0)]);
    let mul1_output = graph.addNode(mul, vec![(sub_output, 0), (less_output, 0)]);
    let check1_output = graph.addNode(sub, vec![(mul1_output, 0), (one_minus_less_output, 0)]);
    let check2_output = graph.addNode(mul, vec![(sub_output, 0), (one_minus_less_output, 0)]);
    let _ = graph.addNode(negative_check, vec![(check1_output, 0)]);
    let _ = graph.addNode(non_negative_check, vec![(check2_output, 0)]);
    graph.outputs.push((less_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}

// Greater layer performs `greater`, an element-wise logical comparison of two tensors.
pub struct GreaterLayer;
impl Layer for GreaterLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    // Inputs: A, B
    // Outputs: L = (A > B); then 1 - L = (A <= B). We can view them as selection of indices.
    // Check 1: (A - B) * L + (1 - L) > 0 because A - B will always > 0 at indices of A > B and we set values at other indices as 1
    // Check 1 is equivalent to (A - B) * L + (1 - L) > 0
    // Check 2: 0 * L + (A - B) * (1 - L) <= 0 because A - B will always <= 0 at indices of A <= B and we set values at other indices as 0
    // Check 2 is equivalent to (A - B) * (1 - L) <= 0
    let mut graph = Graph::new();

    let len = util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize;
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock { len }),
      N: 1,
    }));
    let greater = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(GreaterBasicBlock {}),
      N: 1,
    }));
    let one = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(1); *input_shapes[0].last().unwrap()]).into_dyn(),
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let positive_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: len,
        setup: util::CQArrayType::Positive,
      }),
      N: 1,
    }));
    let non_positive_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: len,
        setup: util::CQArrayType::NonPositive,
      }),
      N: 1,
    }));

    let one_output = graph.addNode(one, vec![]);
    let greater_output = graph.addNode(greater, vec![(-1, 0), (-2, 0)]);
    let one_minus_greater_output = graph.addNode(sub, vec![(one_output, 0), (greater_output, 0)]);
    let sub_output = graph.addNode(sub, vec![(-1, 0), (-2, 0)]);
    let mul1_output = graph.addNode(mul, vec![(sub_output, 0), (greater_output, 0)]);
    let check1_output = graph.addNode(add, vec![(mul1_output, 0), (one_minus_greater_output, 0)]);
    let check2_output = graph.addNode(mul, vec![(sub_output, 0), (one_minus_greater_output, 0)]);
    let _ = graph.addNode(positive_check, vec![(check1_output, 0)]);
    let _ = graph.addNode(non_positive_check, vec![(check2_output, 0)]);
    graph.outputs.push((greater_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}
