use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct MatMulLayer;
impl Layer for MatMulLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let matmul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MatMulBasicBlock {}),
      N: 2,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock { input_SF: 6, output_SF: 3 }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((Box::new(ChangeSFBasicBlock { input_SF: 6, output_SF: 3 }), -(1 << 15), 1 << 16)),
      }),
      N: 1,
    }));
    let matmul_output = graph.addNode(matmul, vec![(-1, 0), (-2, 0)]);
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
    (graph, vec![output_shape])
  }
}
