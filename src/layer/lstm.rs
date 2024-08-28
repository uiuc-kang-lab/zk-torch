use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// reference: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/lstm.py
pub struct LSTMLayer;
impl Layer for LSTMLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    // currently, we do not support P (peepholes)
    assert!(input_shapes.len() == 6); // X, W, R, B, initial_h, initial_c

    let (X_shape, W_shape, _R_shape, B_shape, initial_h_shape, _initial_c_shape) = (
      input_shapes[0],
      input_shapes[1],
      input_shapes[2],
      input_shapes[3],
      input_shapes[4],
      input_shapes[5],
    );
    let (X_index, W_index, R_index, B_index, h_index, c_index) = (-1, -2, -3, -4, -5, -6);

    let seq_length = X_shape[0];
    let hidden_size = initial_h_shape[initial_h_shape.len() - 1];
    let batch_size = X_shape[1]; // currently only supports batch_size = 1
    let num_directions = W_shape[0]; // currently only supports num_directions = 1

    assert!(input_shapes[2][input_shapes[2].len() - 1] == hidden_size);
    assert!(batch_size == 1);
    assert!(num_directions == 1);

    let mut graph = Graph::new();
    // sublayer 1: Split X to X_t
    let split = vec![1; util::next_pow(seq_length as u32) as usize];
    let split_X_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: 0,
      split: split.clone(),
    }));
    let X_output = graph.addNode(split_X_bb, vec![(X_index, 0)]);

    // sublayer 2: Transpose W to W_T (T denotes transpose)
    // but we don't need to do anything here because matmul will handle it
    let W_T_output = W_index;

    // sublayer 3: Transpose R to R_T
    // but we don't need to do anything here because matmul will handle it
    let R_T_output = R_index;

    // sublayer 4: Split for B
    // Here, we need to transpose B first to split it because we cannot split along the last axis
    let axis: usize = 1;
    let split = vec![B_shape[1] / 2, B_shape[1] / 2];

    let mut outputShapes = vec![B_shape.clone(), B_shape.clone()];
    outputShapes[0][1] = split[0];
    outputShapes[1][1] = split[1];

    let n = B_shape.len();
    let mut a = B_shape[n - 2];
    let mut b = B_shape[n - 1];
    (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).map(|x| x).collect());
    let permute = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
      N: 2,
    }));
    let split_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: (axis - 1) as usize,
      split: split.clone().iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
    }));
    let mut permute_backs = vec![];
    for i in 0..split.len() {
      let (mut a, mut b) = (outputShapes[i][n - 2], outputShapes[i][n - 1]);
      (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
      let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
      let permute_back = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock {
          permutation: permutation_back,
        }),
        N: 2,
      }));
      permute_backs.push(permute_back);
    }

    let permute_output = graph.addNode(permute, vec![(B_index, 0)]);
    let split_output = graph.addNode(split_bb, vec![(permute_output, 0)]);
    let mut B_split_output = vec![];
    for i in 0..split.len() {
      let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
      B_split_output.push(output);
    }
    assert!(B_split_output.len() == 2); // we split it into 2 parts

    // sublayer 5: Add splitted B together
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sublayer5 = graph.addNode(add, vec![(B_split_output[0], 0), (B_split_output[1], 0)]);

    // Iterate over t
    let mut H_t_output = h_index;
    let mut C_t_output = c_index;
    let mut H_list = vec![];
    for t in 0..seq_length {
      // sublayer 6: MatMul for X_t and W_T
      let matmul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MatMulBasicBlock {}),
        N: 2,
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
      let matmul_output = graph.addNode(matmul, vec![(X_output, t), (W_T_output, 0)]);
      let sublayer6 = graph.addNode(change_SF, vec![(matmul_output, 0)]); // matmul(X_t, W_T)
      let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (sublayer6, 0)]);

      // sublayer 7: MatMul for H_t and R_T
      let matmul_output = graph.addNode(matmul, vec![(H_t_output, 0), (R_T_output, 0)]);
      let sublayer7 = graph.addNode(change_SF, vec![(matmul_output, 0)]); // matmul(H_t, R_T)
      let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (sublayer7, 0)]);

      // sublayer 8: Calculate matmul(X_t, W_T) + matmul(H_t, R_T)
      let sublayer8 = graph.addNode(add, vec![(sublayer6, 0), (sublayer7, 0)]);

      // sublayer 9: Calculate gates = matmul(X_t, W_T) + matmul(H_t, R_T) + add(*B.split())
      let gates = graph.addNode(add, vec![(sublayer8, 0), (sublayer5, 0)]);

      // sublayer 10: Split gates into input_gate, output_gate, forget_gate, candidate_memory
      // Here, we need to transpose gates first to split it because we cannot split along the last axis
      let axis: usize = 2;
      let split = vec![W_shape[1] / 4; 4];
      assert!(split[0] == hidden_size);

      let outputShapes = vec![vec![1, batch_size, hidden_size]; 4];

      let outputShape = outputShapes[0].clone();
      let n = outputShape.len();
      let mut a = outputShape[n - 2];
      let mut b = 4 * outputShape[n - 1];
      (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
      let permutation = ((0..b).map(|x| x * a).collect(), (0..a).map(|x| x).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let split_bb = graph.addBB(Box::new(SplitBasicBlock {
        axis: (axis - 1) as usize,
        split: split.clone().iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let mut permute_backs = vec![];
      for i in 0..split.len() {
        let (mut a, mut b) = (outputShapes[i][n - 2], outputShapes[i][n - 1]);
        (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
        let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
        let permute_back = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(PermuteBasicBlock {
            permutation: permutation_back,
          }),
          N: 2,
        }));
        permute_backs.push(permute_back);
      }

      let permute_output = graph.addNode(permute, vec![(gates, 0)]);
      let split_output = graph.addNode(split_bb, vec![(permute_output, 0)]);
      let mut gates = vec![];
      for i in 0..split.len() {
        let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
        gates.push(output);
      }

      // sublayer 11: Sigmoid for input gate
      let sigmoid = graph.addBB(Box::new(SigmoidBasicBlock {
        input_SF: *onnx::SF_LOG * 2,
        output_SF: *onnx::SF_LOG,
      }));
      let sigmoid_check = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQ2BasicBlock {
          setup: Some((
            Box::new(SigmoidBasicBlock {
              input_SF: *onnx::SF_LOG * 2,
              output_SF: *onnx::SF_LOG,
            }),
            *onnx::CQ_RANGE_LOWER,
            *onnx::CQ_RANGE,
          )),
        }),
        N: 1,
      }));
      let input_gate_output = graph.addNode(sigmoid, vec![(gates[0], 0)]);
      let _ = graph.addNode(sigmoid_check, vec![(gates[0], 0), (input_gate_output, 0)]);

      // sublayer 12: Sigmoid for output gate
      let output_gate_output = graph.addNode(sigmoid, vec![(gates[1], 0)]);
      let _ = graph.addNode(sigmoid_check, vec![(gates[1], 0), (output_gate_output, 0)]);

      // sublayer 13: Sigmoid for forget gate
      let forget_gate_output = graph.addNode(sigmoid, vec![(gates[2], 0)]);
      let _ = graph.addNode(sigmoid_check, vec![(gates[2], 0), (forget_gate_output, 0)]);

      // sublayer 14: Tanh for candidate memory
      let tanh = graph.addBB(Box::new(TanhBasicBlock {
        input_SF: *onnx::SF_LOG * 2,
        output_SF: *onnx::SF_LOG,
      }));
      let tanh_check = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQ2BasicBlock {
          setup: Some((
            Box::new(TanhBasicBlock {
              input_SF: *onnx::SF_LOG * 2,
              output_SF: *onnx::SF_LOG,
            }),
            *onnx::CQ_RANGE_LOWER,
            *onnx::CQ_RANGE,
          )),
        }),
        N: 1,
      }));
      let candidate_memory_output = graph.addNode(tanh, vec![(gates[3], 0)]);
      let _ = graph.addNode(tanh_check, vec![(gates[3], 0), (candidate_memory_output, 0)]);

      // sublayer 15: input_gate * candidate_memory
      let mul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulBasicBlock {}),
        N: 1,
      }));
      let mul_output = graph.addNode(mul, vec![(input_gate_output, 0), (candidate_memory_output, 0)]);
      let sublayer15 = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (sublayer15, 0)]);

      // sublayer 16: forget gate * C_t
      let mul_output = graph.addNode(mul, vec![(forget_gate_output, 0), (C_t_output, 0)]);
      let sublayer16 = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (sublayer16, 0)]);

      // sublayer 17: C = sublayer15 + sublayer16 = input_gate * candidate_memory + forget gate * C_t
      let C = graph.addNode(add, vec![(sublayer15, 0), (sublayer16, 0)]);

      // sublayer 18: Tanh(C)
      let sublayer18 = graph.addNode(tanh, vec![(C, 0)]);
      let _ = graph.addNode(tanh_check, vec![(C, 0), (sublayer18, 0)]);

      // sublayer 19: H = output_gate * Tanh(C)
      let mul_output = graph.addNode(mul, vec![(output_gate_output, 0), (sublayer18, 0)]);
      let H = graph.addNode(change_SF, vec![(mul_output, 0)]); // shape = [num_directions, batch_size, hidden_size]
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (H, 0)]);

      // update H_t_output and C_t_output
      H_t_output = H;
      C_t_output = C;

      // sublayer 20: Unsqueeze H at axis 0
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock {
        shape: vec![1, num_directions, batch_size, hidden_size].iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let H_t = graph.addNode(reshape, vec![(H, 0)]);
      H_list.push(H_t);
    }

    for _t in 0..util::next_pow(seq_length as u32) as usize - seq_length {
      // sublayer 21: Pad H_t with zeros
      let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::zero(),
        shape: vec![1, num_directions, batch_size, hidden_size].iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let H_t_pad = graph.addNode(constantOfShape, vec![]);
      H_list.push(H_t_pad);
    }

    // sublayer 22: Concat H_list
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 0 }));
    let output = graph.addNode(concat, H_list.iter().map(|&H_t| (H_t, 0)).collect());

    graph.outputs.push((output, 0)); // Y
    graph.outputs.push((H_list[H_list.len() - 1], 0)); // Y_h
    graph.outputs.push((H_list[H_list.len() - 1], 0)); // Y_c

    (
      graph,
      vec![
        vec![seq_length, num_directions, batch_size, hidden_size],
        vec![num_directions, batch_size, hidden_size],
        vec![num_directions, batch_size, hidden_size],
      ],
      vec![input_types[0]; 3],
    )
  }
}
