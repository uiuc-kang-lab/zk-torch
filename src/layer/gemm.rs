use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Gemm computes Y = alpha * A' * B' + beta * C, where
// the first input tensor A has shape (M, K) or (K, M),
// the second input tensor B has shape (K, N) or (N, K),
// (optional) the third input tensor C is broadcastable to shape (M, N),
// and output tensor Y has shape (M, N).
// A will be transposed to A' before doing the computation if attribute transA is non-zero, same for B and transB.
pub struct GemmLayer;
impl Layer for GemmLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let alpha = if attributes.iter().any(|x| x.name == "alpha") {
      attributes.iter().filter(|x| x.name == "alpha").next().unwrap().f
    } else {
      1.0
    };
    let beta = if attributes.iter().any(|x| x.name == "beta") {
      attributes.iter().filter(|x| x.name == "beta").next().unwrap().f
    } else {
      1.0
    };
    let transA = if attributes.iter().any(|x| x.name == "transA") {
      attributes.iter().filter(|x| x.name == "transA").next().unwrap().i as usize
    } else {
      0
    };
    let transB = if attributes.iter().any(|x| x.name == "transB") {
      attributes.iter().filter(|x| x.name == "transB").next().unwrap().i as usize
    } else {
      0
    };

    let (M, K_a) = if transA == 0 {
      (input_shapes[0][0], input_shapes[0][1])
    } else {
      (input_shapes[0][1], input_shapes[0][0])
    };
    let (K_b, N) = if transB == 0 {
      (input_shapes[1][0], input_shapes[1][1])
    } else {
      (input_shapes[1][1], input_shapes[1][0])
    };
    assert!(K_a == K_b);

    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
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
    let alpha = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from((alpha * *onnx::SF_FLOAT) as i64)]).into_dyn(),
    }));
    let beta = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from((beta * *onnx::SF_FLOAT) as i64)]).into_dyn(),
    }));

    let M_pad = util::next_pow(M as u32) as usize;
    let N_pad = util::next_pow(N as u32) as usize;
    let K_pad = util::next_pow(K_a as u32) as usize;
    let permutation_A = ((0..M_pad).map(|x| x * K_pad).collect(), (0..K_pad).collect());
    let permutation_B = ((0..N_pad).map(|x| x * K_pad).collect(), (0..K_pad).collect());
    let mut A_output = -1;
    let mut B_output = -2;
    if transA != 0 {
      let transpose_A = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_A }),
        N: 2,
      }));
      A_output = graph.addNode(transpose_A, vec![(A_output, 0)]);
    }
    if transB == 0 {
      let transpose_B = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_B }),
        N: 2,
      }));
      B_output = graph.addNode(transpose_B, vec![(B_output, 0)]);
    }

    let alpha_output = graph.addNode(alpha, vec![]);
    let matmul_output = graph.addNode(matmul, vec![(A_output, 0), (B_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(matmul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(matmul_output, 0), (change_SF_output, 0)]);
    let mul_output_AB = graph.addNode(mul_scalar, vec![(change_SF_output, 0), (alpha_output, 0)]);
    let mut output = graph.addNode(change_SF, vec![(mul_output_AB, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output_AB, 0), (change_SF_output, 0)]);
    if input_shapes.len() > 2 {
      // C exists
      let beta_output = graph.addNode(beta, vec![]);
      let mul_output_C = graph.addNode(mul_scalar, vec![(-3, 0), (beta_output, 0)]);
      let change_SF_output_C = graph.addNode(change_SF, vec![(mul_output_C, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output_C, 0), (change_SF_output_C, 0)]);
      output = graph.addNode(add, vec![(output, 0), (change_SF_output_C, 0)]);
    }

    graph.outputs.push((output, 0));

    let output_shape = vec![M, N];
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
