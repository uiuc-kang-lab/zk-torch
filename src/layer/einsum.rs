use crate::basic_block::*;
use crate::graph::*;
use crate::layer::{squeeze::UnsqueezeBasicBlock, Layer};
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use std::collections::HashMap;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

fn parse_einsum(equation: &str) -> (Vec<String>, Vec<String>) {
  // Helper function to map letters to a standard set while preserving "..."
  fn standardize(input: &str, map: &mut HashMap<char, char>, current: &mut char) -> String {
    let mut result = String::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
      if ch == '.' && chars.clone().take(3).collect::<String>() == "..." {
        result.push_str("...");
        for _ in 0..3 {
          chars.next();
        }
      } else {
        if !map.contains_key(&ch) {
          map.insert(ch, *current);
          *current = ((*current as u8) + 1) as char;
        }
        result.push(*map.get(&ch).unwrap());
        chars.next();
      }
    }
    result
  }

  // Remove all spaces from the equation
  let equation = equation.replace(" ", "");

  // Split the equation into left-hand side (lhs) and right-hand side (rhs)
  let parts: Vec<&str> = equation.split("->").collect();

  // Split the lhs into individual operands
  let lhs: Vec<&str> = parts[0].split(',').collect();

  // Split the rhs into the resulting dimensions, if present
  let rhs: Vec<&str> = if parts.len() > 1 {
    parts[1].split_whitespace().collect()
  } else {
    Vec::new()
  };

  // Create a standardized mapping for the letters
  let mut map = HashMap::new();
  let mut current = 'a';
  let lhs_standardized: Vec<String> = lhs.iter().map(|&s| standardize(s, &mut map, &mut current)).collect();
  let rhs_standardized: Vec<String> = rhs.iter().map(|&s| standardize(s, &mut map, &mut current)).collect();

  (lhs_standardized, rhs_standardized)
}

fn vector_outer_product(graph: &mut Graph, input_shapes: &Vec<&Vec<usize>>) -> Vec<usize> {
  let unsqueeze = graph.addBB(Box::new(UnsqueezeBasicBlock {}));
  let to_split = vec![1; util::next_pow(input_shapes[0][0] as u32) as usize];
  let split = graph.addBB(Box::new(SplitBasicBlock { axis: 0, split: to_split }));
  let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(MulScalarBasicBlock {}),
    N: 1,
  }));
  let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
    input_SF: *onnx::SF_LOG * 2,
    output_SF: *onnx::SF_LOG,
  }));
  let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(CQ2BasicBlock {
      setup: Some((
        Box::new(ChangeSFBasicBlock {
          input_SF: *onnx::SF_LOG * 2,
          output_SF: *onnx::SF_LOG,
        }),
        *onnx::CQ_RANGE_LOWER,
        *onnx::CQ_RANGE,
      )),
    }),
    N: 1,
  }));
  let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 0 }));
  let mut b = input_shapes[0][0];
  b = util::next_pow(b as u32) as usize;
  let permutation = ((0..b).map(|x| x).collect(), vec![0]);
  let permute = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
    N: 2,
  }));
  // to split the input vector into scalars, we first unsqueeze and permute it
  let unsqueeze_output = graph.addNode(unsqueeze, vec![(-1, 0)]);
  let permute_output = graph.addNode(permute, vec![(unsqueeze_output, 0)]);
  let split_output = graph.addNode(split, vec![(permute_output, 0)]);
  // for each scalar, we multiply it by the other input vector
  let mut mul_scalar_outputs = vec![];
  for i in 0..input_shapes[0][0] {
    let mul_scalar_output = graph.addNode(mul_scalar, vec![(-2, 0), (split_output, i)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_scalar_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_scalar_output, 0), (change_SF_output, 0)]);
    mul_scalar_outputs.push(change_SF_output);
  }
  // finally, we concatenate the results to get the outer product
  let concat_output = graph.addNode(concat, mul_scalar_outputs.iter().map(|x| (*x, 0)).collect());
  let output_shape = vec![input_shapes[0][0], input_shapes[1][0]];
  graph.outputs.push((concat_output, 0));
  output_shape
}

fn vector_inner_product(graph: &mut Graph, _input_shapes: &Vec<&Vec<usize>>) -> Vec<usize> {
  let mul = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(MulBasicBlock {}),
    N: 1,
  }));
  let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
    input_SF: *onnx::SF_LOG * 2,
    output_SF: *onnx::SF_LOG,
  }));
  let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(CQ2BasicBlock {
      setup: Some((
        Box::new(ChangeSFBasicBlock {
          input_SF: *onnx::SF_LOG * 2,
          output_SF: *onnx::SF_LOG,
        }),
        *onnx::CQ_RANGE_LOWER,
        *onnx::CQ_RANGE,
      )),
    }),
    N: 1,
  }));
  let sum = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(SumBasicBlock {}),
    N: 1,
  }));
  let mul_output = graph.addNode(mul, vec![(-1, 0), (-2, 0)]);
  let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
  let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
  let sum_output = graph.addNode(sum, vec![(change_SF_output, 0)]);
  let output_shape = vec![1];
  graph.outputs.push((sum_output, 0));
  output_shape
}

// EinsumLayer implements the einsum operation in ONNX. It supports the following equations:
// - "a,b->ab" (vector outer product)
// - "a,a->" (vector inner product)
// Other equations are not supported yet.
pub struct EinsumLayer;
impl Layer for EinsumLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let equation = &attributes.iter().filter(|x| x.name == "equation").next().unwrap().s;
    let equation = std::str::from_utf8(&equation).unwrap();
    let (input_eqs, output_eq) = parse_einsum(equation);
    assert!(input_eqs.len() == input_shapes.len());
    // vector outer product
    if input_eqs == vec!["a", "b"] && output_eq == vec!["ab"] {
      assert!(input_shapes[0].len() == 1 && input_shapes[1].len() == 1);
      let output_shape = vector_outer_product(&mut graph, input_shapes);
      (graph, vec![output_shape], vec![input_types[0]])
    // vector inner product
    } else if input_eqs == vec!["a", "a"] && output_eq.len() == 0 {
      assert!(input_shapes[0].len() == 1 && input_shapes[1].len() == 1);
      let output_shape = vector_inner_product(&mut graph, input_shapes);
      (graph, vec![output_shape], vec![input_types[0]])
    } else {
      panic!("EinsumLayer not implemented for equation {:?}", equation);
    }
  }
}
