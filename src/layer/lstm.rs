use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct LSTMLayer;
impl Layer for LSTMLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
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

    let seq_length = X_shape[0]; // currently only supports seq_length = 1
    let hidden_size = initial_h_shape[initial_h_shape.len() - 1];
    let batch_size = X_shape[1]; // currently only supports batch_size = 1
    let num_directions = W_shape[0]; // currently only supports num_directions = 1
    assert!(seq_length == 1);
    assert!(input_shapes[2][input_shapes[2].len() - 1] == hidden_size);
    assert!(batch_size == 1);
    assert!(num_directions == 1);
    /* reference: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/lstm.py
      suppose X is of shape (seq_length, batch_size, feature_dim); but seq_length is always 1 for now
       for x in np.split(self.X, self.X.shape[0], axis=0):
           gates = (
               np.dot(x, np.transpose(self.W))
               + np.dot(H_t, np.transpose(self.R))
               + np.add(*np.split(self.B, 2))
           )
           i, o, f, c = np.split(gates, 4, -1)
           i = self.f(i)
           o = self.f(o)
           f = self.f(f)
           c = self.g(c)
           C = f * C_t + i * c

           H = o * self.h(C)
           h_list.append(H)
           H_t = H
           C_t = C

       concatenated = np.concatenate(h_list)
       Y[:, 0, :, :] = concatenated
       Y_h = Y[-1]
       return Y, Y_h
    */

    let mut graph = Graph::new();
    // sublayer 1: MatMul for X and W
    let n = input_shapes[1].len();
    let (mut a, mut b) = (input_shapes[1][n - 2], input_shapes[1][n - 1]);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    // let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    // let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
    //   basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
    //   N: 2,
    // }));
    let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: vec![a, b] }));

    let matmul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MatMulBasicBlock {}),
      N: 2,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: onnx::SF_LOG * 2,
      output_SF: onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: onnx::SF_LOG * 2,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let reshape_output = graph.addNode(reshape, vec![(W_index, 0)]);
    let matmul_output = graph.addNode(matmul, vec![(X_index, 0), (reshape_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(matmul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (change_SF_output, 0)]);
    let sublayer1 = change_SF_output;

    // sublayer 2: MatMul for initial_h and R
    let n = input_shapes[2].len();
    let (mut a, mut b) = (input_shapes[2][n - 2], input_shapes[2][n - 1]);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;

    // let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    // let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
    //   basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
    //   N: 2,
    // }));
    let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: vec![a, b] }));
    let matmul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MatMulBasicBlock {}),
      N: 2,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: onnx::SF_LOG * 2,
      output_SF: onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: onnx::SF_LOG * 2,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let reshape_output = graph.addNode(reshape, vec![(R_index, 0)]);
    let matmul_output = graph.addNode(matmul, vec![(h_index, 0), (reshape_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(matmul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (change_SF_output, 0)]);
    let sublayer2 = change_SF_output;

    // sublayer 3: Split for B
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
      split: split.clone(),
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
    let mut sublayer3 = vec![];
    for i in 0..split.len() {
      let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
      sublayer3.push(output);
    }

    // sublayer 4: Add for s1 and s2
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let add_output = graph.addNode(add, vec![(sublayer1, 0), (sublayer2, 0)]);
    let sublayer4 = add_output;

    // sublayer 5: Add for s3
    let add_output = graph.addNode(add, vec![(sublayer3[0], 0), (sublayer3[1], 0)]);
    let sublayer5 = add_output;

    // sublayer 6: Add for s4 and s5
    let add_output = graph.addNode(add, vec![(sublayer4, 0), (sublayer5, 0)]);
    let sublayer6 = add_output;

    // sublayer 7: Split for s6 to s7.1, s7.2, s7.3, s7.4
    let axis: usize = 2;
    let split = vec![W_shape[1] / 4; 4];

    let outputShapes = vec![vec![X_shape[0], X_shape[1], split[0]]; 4];

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
      split: split.clone(),
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

    let permute_output = graph.addNode(permute, vec![(sublayer6, 0)]);
    let split_output = graph.addNode(split_bb, vec![(permute_output, 0)]);
    let mut sublayer7 = vec![];
    for i in 0..split.len() {
      let output = graph.addNode(permute_backs[i], vec![(split_output, i)]);
      sublayer7.push(output);
    }

    // sublayer 8: Sigmoid for s7.1 (i)
    let sigmoid = graph.addBB(Box::new(SigmoidBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let sigmoid_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(SigmoidBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sigmoid_output = graph.addNode(sigmoid, vec![(sublayer7[0], 0)]);
    let _ = graph.addNode(sigmoid_check, vec![(sublayer7[0], 0), (sigmoid_output, 0)]);
    let sublayer8 = sigmoid_output;

    // sublayer 9: Sigmoid for s7.2 (o)
    let sigmoid_output = graph.addNode(sigmoid, vec![(sublayer7[1], 0)]);
    let _ = graph.addNode(sigmoid_check, vec![(sublayer7[1], 0), (sigmoid_output, 0)]);
    let sublayer9 = sigmoid_output;

    // sublayer 10: Sigmoid for s7.3 (f)
    let sigmoid_output = graph.addNode(sigmoid, vec![(sublayer7[2], 0)]);
    let _ = graph.addNode(sigmoid_check, vec![(sublayer7[2], 0), (sigmoid_output, 0)]);
    let sublayer10 = sigmoid_output;

    // sublayer 11: Tanh for s7.4 (c)
    let tanh = graph.addBB(Box::new(TanhBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let tanh_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(TanhBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let tanh_output = graph.addNode(tanh, vec![(sublayer7[3], 0)]);
    let _ = graph.addNode(tanh_check, vec![(sublayer7[3], 0), (tanh_output, 0)]);
    let sublayer11 = tanh_output;

    // sublayer 12: Multiply for s8 (i) and s11 (c)
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: onnx::SF_LOG * 2,
      output_SF: onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: onnx::SF_LOG * 2,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let mul_output = graph.addNode(mul, vec![(sublayer8, 0), (sublayer11, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    let sublayer12 = change_SF_output;

    // sublayer 13: Multiply for s10 (f) and initial_c
    let mul_output = graph.addNode(mul, vec![(sublayer10, 0), (c_index, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    let sublayer13 = change_SF_output;

    // sublayer 14: Add for s12 and s13
    let add_output = graph.addNode(add, vec![(sublayer12, 0), (sublayer13, 0)]);
    let sublayer14 = add_output;

    // sublayer 15: Tanh for s14
    let tanh_output = graph.addNode(tanh, vec![(sublayer14, 0)]);
    let _ = graph.addNode(tanh_check, vec![(sublayer14, 0), (tanh_output, 0)]);
    let sublayer15 = tanh_output;

    // sublayer 16: Multiply for s9 (o) and s15
    let mul_output = graph.addNode(mul, vec![(sublayer9, 0), (sublayer15, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    let sublayer16 = change_SF_output;

    // reshape s16 to get Y, Y_h (s16), Y_c (s16)
    let reshape = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![1, 1, 1, hidden_size],
    }));
    let output = graph.addNode(reshape, vec![(sublayer16, 0)]);

    graph.outputs.push((output, 0)); // Y
    graph.outputs.push((sublayer16, 0)); // Y_h
    graph.outputs.push((sublayer16, 0)); // Y_c

    (graph, vec![vec![1, 1, 1, hidden_size], vec![1, 1, hidden_size], vec![1, 1, hidden_size]])
  }
}
