use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct MatMulLayer;
impl Layer for MatMulLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let n = input_shapes[1].len();
    let (mut a, mut b) = (input_shapes[1][n - 2], input_shapes[1][n - 1]);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());

    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
      N: 2,
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
    let transpose_output = graph.addNode(transpose, vec![(-2, 0)]);
    let matmul_output = graph.addNode(matmul, vec![(-1, 0), (transpose_output, 0)]);
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
