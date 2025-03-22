use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::One;
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
    let mul_scalar = if input_shapes[0].len() == input_shapes[1].len() && input_shapes[0].len() == 0 {
      graph.addBB(Box::new(MulScalarBasicBlock {}))
    } else {
      graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulScalarBasicBlock {}),
        N: 1,
      }))
    };
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = if input_shapes[0].len() == input_shapes[1].len() && input_shapes[0].len() == 0 {
      graph.addBB(Box::new(CQ2BasicBlock {
        n: 1,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }))
    } else {
      graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQ2BasicBlock {
          n: if input_shapes[1].len() == 0 {
            input_shapes[0][input_shapes[0].len() - 1].next_power_of_two()
          } else {
            std::cmp::max(
              input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
              input_shapes[1][input_shapes[1].len() - 1].next_power_of_two(),
            )
          },
          setup: Some((
            Box::new(ChangeSFBasicBlock {
              input_SF: sf_log * 2,
              output_SF: sf_log,
            }),
            *onnx::CQ_RANGE_LOWER,
            *onnx::CQ_RANGE,
          )),
        }),
        N: 1,
      }))
    };
    // If any of the inputs are scalars, use the scalar version of the mul basic block.
    // If the first input is a scalar, swap the inputs, because the mul scalar basic block expects the scalar to be the second input. If the last dimension differs between the two inputs, broadcast.
    let mul_output = if input_shapes[0].len() == 0 {
      graph.addNode(mul_scalar, vec![(-2, 0), (-1, 0)])
    } else if input_shapes[1].len() > 0 && input_shapes[0].last().unwrap() != input_shapes[1].last().unwrap() {
      let (broadcast_inp, mul_inp, broadcast_idx) = if input_shapes[0].last().unwrap() > input_shapes[1].last().unwrap() {
        (-2, -1, 0)
      } else {
        (-1, -2, 1)
      };
      let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::one(),
        shape: input_shapes[broadcast_idx].clone(),
      }));
      let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulScalarBasicBlock {}),
        N: 1,
      }));
      let constantOfShape_output = graph.addNode(constantOfShape, vec![]);
      if *input_shapes[0].last().unwrap() == 1 || *input_shapes[1].last().unwrap() == 1 {
        let broadcast_output = graph.addNode(mul_scalar, vec![(constantOfShape_output, 0), (broadcast_inp, 0)]);
        let mul_inp_idx = (-mul_inp - 1) as usize;
        let len = util::next_pow(input_shapes[mul_inp_idx][input_shapes[mul_inp_idx].len() - 1] as u32) as usize;
        let mul = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(MulBasicBlock { len }),
          N: 1,
        }));
        graph.addNode(mul, vec![(mul_inp, 0), (broadcast_output, 0)])
      } else {
        let inp_shape = input_shapes[broadcast_idx];
        let len = util::next_pow(inp_shape[inp_shape.len() - 1] as u32) as usize;
        let mul = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(MulBasicBlock { len }),
          N: 1,
        }));
        let broadcast_output = graph.addNode(mul, vec![(constantOfShape_output, 0), (broadcast_inp, 0)]);
        graph.addNode(mul, vec![(broadcast_output, 0), (mul_inp, 0)])
      }
    } else {
      let mul_basicblock = if input_shapes[1].len() == 0 || input_shapes[0].len() == 0 {
        mul_scalar
      } else {
        let len = if input_shapes[0].len() > input_shapes[1].len() {
          util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize
        } else {
          util::next_pow(input_shapes[1][input_shapes[1].len() - 1] as u32) as usize
        };
        graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(MulBasicBlock { len }),
          N: 1,
        }))
      };
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
