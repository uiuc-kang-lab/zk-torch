use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct MulLayer;
impl Layer for MulLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let mul_scalar = if input_shapes[0].len() == input_shapes[1].len() && input_shapes[0].len() == 0 {
      graph.addBB(Box::new(MulScalarBasicBlock {}))
    } else {
      graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulScalarBasicBlock {}),
        N: 1,
      }))
    };
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: *onnx::SF_LOG * 2,
      output_SF: *onnx::SF_LOG,
    }));
    let change_SF_check = if input_shapes[0].len() == input_shapes[1].len() && input_shapes[0].len() == 0 {
      graph.addBB(Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: *onnx::SF_LOG * 2,
            output_SF: *onnx::SF_LOG,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }))
    } else {
      graph.addBB(Box::new(RepeaterBasicBlock {
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
      }))
    };
    // If any of the inputs are scalars, use the scalar version of the mul basic block.
    let mul_basicblock = if input_shapes[1].len() == 0 || input_shapes[0].len() == 0 {
      mul_scalar
    // otherwise, use the normal version of the mul basic block.
    } else {
      mul
    };
    // If the first input is a scalar, swap the inputs, because the mul scalar basic block expects the scalar to be the second input.
    let mul_output = if input_shapes[0].len() == 0 {
      graph.addNode(mul_basicblock, vec![(-2, 0), (-1, 0)])
    } else {
      graph.addNode(mul_basicblock, vec![(-1, 0), (-2, 0)])
    };

    if input_types[0].is_float() {
      let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
      graph.outputs.push((change_SF_output, 0));
    } else if input_types[0].is_integer() {
      graph.outputs.push((mul_output, 0));
    } else {
      panic!("Mul input type {:?} is not supported", input_types[0]);
    }
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![input_types[0]])
  }
}
