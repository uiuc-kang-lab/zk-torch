use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{Array1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Define a macro to unify common behavior for DivLayer and ModLayer
// when output_idx equals to 0, the layer returns the quotient. (Div)
// when output_idx equals to 1, the layer returns the remainder. (Mod)
macro_rules! create_division_layer {
  ($layer_name:ident, $const_block:ident, $output_idx:expr) => {
    pub struct $layer_name;

    impl Layer for $layer_name {
      fn graph(
        input_shapes: &Vec<&Vec<usize>>, // Input shapes for the layer
        input_types: &Vec<DatumType>,
        constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>, // Constants used in the layer
        _attributes: &Vec<&AttributeProto>,                // Attributes for the layer (not used in this implementation)
      ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
        let mut graph = Graph::new(); // Create a new computational graph

        // Check if the second input is a constant
        if let Some(c) = constants[1] {
          // Assert the length of the constant is 1
          assert!(c.0.len() == 1);

          // Convert the constant to a floating-point number
          let c_value = util::fr_to_int(*c.0.first().unwrap()) as f32;
          let c_value = match c.1 {
            DatumType::I64 => c_value,
            _ => {
              if $output_idx == 0 {
                c_value / *onnx::SF_FLOAT as f32
              } else {
                c_value
              }
            }
          };

          // Add a basic block for division/modulo by a constant
          let const_block = graph.addBB(Box::new($const_block { c: c_value as _ }));

          // Add a basic block for range checking with custom setup
          let const_check = if input_shapes[0].len() == input_shapes[1].len() && input_shapes[0].len() == 0 {
            graph.addBB(Box::new(CQ2BasicBlock {
              setup: Some((Box::new($const_block { c: c_value as _ }), *onnx::CQ_RANGE_LOWER, *onnx::CQ_RANGE)),
            }))
          } else {
            graph.addBB(Box::new(RepeaterBasicBlock {
              basic_block: Box::new(CQ2BasicBlock {
                setup: Some((Box::new($const_block { c: c_value as _ }), *onnx::CQ_RANGE_LOWER, *onnx::CQ_RANGE)),
              }),
              N: 1,
            }))
          };

          // Create a node for the division/modulo operation
          let const_output = graph.addNode(const_block, vec![(-1, 0)]);

          // Add a node for range checking, ensuring it processes the division/modulo output
          let _ = graph.addNode(const_check, vec![(-1, 0), (const_output, 0)]);

          // Set the output of the graph
          graph.outputs.push((const_output, 0));

          // Return the graph and updated input shapes
          return (graph, vec![input_shapes[0].clone()], vec![input_types[0]]);
        }

        // Assert that the second input has only one element
        assert!(*input_shapes[1].last().unwrap() == 1);

        // Create a basic block for division with scalar values
        let div = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(DivScalarBasicBlock { output_SF: *onnx::SF }),
          N: 1,
        }));

        // Add a range check basic block for ensuring the remainder is non-negative
        let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQBasicBlock {
            setup: Array1::from_iter(0..-*onnx::CQ_RANGE_LOWER).map(|x| Fr::from(*x as i32)),
          }),
          N: 1,
        }));

        // Create basic blocks for multiplication by constants and scalars
        let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(MulConstBasicBlock { c: *onnx::SF * 2 }),
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

        // Add basic blocks for addition, subtraction, and equality checks
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

        // a is the dividend (first input)
        // b is the divisor (second input)
        // q and r are the quotient and remainder for (2*a*SF + b) / (2 * b)

        // Calculate the division/modulo output node
        let div_output = graph.addNode(div, vec![(-1, 0), (-2, 0)]);

        // Compute 2 * a * SF
        let a_SF2 = graph.addNode(mul_SF2, vec![(-1, 0)]);

        // Compute (2 * a * SF) + b
        let a_SF2_plus_b = graph.addNode(add, vec![(a_SF2, 0), (-2, 0)]);

        // Compute 2 * b
        let b2 = graph.addNode(mul_2, vec![(-2, 0)]);

        // Calculate q * b * 2
        let qb2 = graph.addNode(mul, vec![(div_output, 0), (b2, 0)]);

        // Add q * b * 2 and the remainder r
        let qb2_plus_r = graph.addNode(add, vec![(qb2, 0), (div_output, 1)]);

        // Ensure that (2 * a * SF) + b equals q * b * 2 + r
        let _ = graph.addNode(eq, vec![(a_SF2_plus_b, 0), (qb2_plus_r, 0)]);

        // Check if the remainder r is non-negative
        let _ = graph.addNode(range_check, vec![(div_output, 1)]);

        // Check if 2 * b - r is non-negative
        let b2_minus_r = graph.addNode(sub, vec![(b2, 0), (div_output, 1)]);
        let _ = graph.addNode(range_check, vec![(b2_minus_r, 0)]);

        // Set the output node for the graph
        graph.outputs.push((div_output, $output_idx));

        // Return the constructed graph and the updated input shapes
        (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
      }
    }
  };
}

// Create DivLayer and ModLayer using the macro
create_division_layer!(DivLayer, DivConstBasicBlock, 0);
create_division_layer!(ModLayer, ModConstBasicBlock, 1);
