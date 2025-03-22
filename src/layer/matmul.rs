use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::CONFIG;
use ark_bn254::Fr;
use ndarray::{s, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct MatMulLayer;
impl Layer for MatMulLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let n = input_shapes[1].len();
    let (mut a, mut b) = (input_shapes[1][n - 2], input_shapes[1][n - 1]);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());

    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: input_shapes[1][input_shapes[1].len() - 1].next_power_of_two(),
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
    }));
    let small_matmul = (a * b) < 1 << CONFIG.ptau.loaded_pow_len_log;
    let use_cqlin = constants.len() > 1 && constants[1].is_some() && small_matmul;
    let matmul_output = if use_cqlin {
      let b = constants[1].unwrap().0;
      let cqlin = if input_shapes[0].len() > 1 {
        graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQLinBasicBlock { setup: b.to_owned() }),
          N: 2,
        }))
      } else {
        graph.addBB(Box::new(CQLinBasicBlock { setup: b.to_owned() }))
      };
      graph.addNode(cqlin, vec![(-1, 0)])
    } else {
      let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock {
          permutation: permutation,
          n: a,
          m: b,
        }),
        N: 2,
      }));
      let matmul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MatMulBasicBlock { m: a, n: b }),
        N: 2,
      }));
      let transpose_output = graph.addNode(transpose, vec![(-2, 0)]);
      graph.addNode(matmul, vec![(-1, 0), (transpose_output, 0)])
    };
    let change_SF_output = graph.addNode(change_SF, vec![(matmul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (change_SF_output, 0)]);
    graph.outputs.push((change_SF_output, 0));

    let mut output_shape = util::broadcastDims(input_shapes, 2);
    if input_shapes[0].len() >= 2 {
      output_shape.push(input_shapes[0][input_shapes[0].len() - 2]);
      output_shape.push(input_shapes[1][input_shapes[1].len() - 1]);
    } else {
      output_shape.push(input_shapes[1][input_shapes[1].len() - 1]);
    }
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

// Assume the shape of input tensor t is [batch, sequence, f] and that of matrix A is [f, f]
// Rather than compute s = A*t and Reshape(s, [batch, sequence, p, q]) (p*q=f) <-- this one is expensive
// We compute the following:
// 1. Precut the matrix A at compilation time into A_0, A_1, ..., A_p, each of which is f x q
// 2. Compute s_i = A_i*t for all i in 0..p
// 3. Unsqueeze and concat s_i at axis 2 (these operations are free because they are not run at the last axis)
pub struct MultiHeadMatMulLayer;
impl Layer for MultiHeadMatMulLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    assert!(constants[1].is_some() && constants[2].is_some());
    let B = constants[1].unwrap().0;
    let output_shape: Vec<_> = constants[2].unwrap().0.iter().map(|x| util::fr_to_int(*x) as usize).collect();

    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: output_shape[output_shape.len() - 1].next_power_of_two(),
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
    }));

    let mut unsq_shape = output_shape.clone();
    unsq_shape[output_shape.len() - 2] = 1;
    let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: unsq_shape.clone() }));
    let mut inputs_to_concat = vec![];
    for i in 0..output_shape[output_shape.len() - 2] {
      let start_col = i * output_shape[output_shape.len() - 1];
      let end_col = start_col + output_shape[output_shape.len() - 1];

      let b = B.slice(s![.., start_col..end_col]);
      let cqlin = if input_shapes[0].len() > 1 {
        graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQLinBasicBlock {
            setup: b.to_owned().into_dyn(),
          }),
          N: 2,
        }))
      } else {
        graph.addBB(Box::new(CQLinBasicBlock {
          setup: b.to_owned().into_dyn(),
        }))
      };
      let one_head_output = graph.addNode(cqlin, vec![(-1, 0)]);
      let unsq = graph.addNode(reshape, vec![(one_head_output, 0)]);
      inputs_to_concat.push((unsq, 0));
    }
    let concat = graph.addBB(Box::new(ConcatBasicBlock {
      axis: output_shape.len() - 2,
    }));
    let multihead_output = graph.addNode(concat, inputs_to_concat);
    let change_SF_output = graph.addNode(change_SF, vec![(multihead_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(multihead_output, 0), (change_SF_output, 0)]);
    graph.outputs.push((change_SF_output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
