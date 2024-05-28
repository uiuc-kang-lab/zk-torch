use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct SqrtLayer;
impl Layer for SqrtLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let sqrt_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(SqrtBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sqrt_output = graph.addNode(sqrt, vec![(-1, 0)]);
    let _ = graph.addNode(sqrt_check, vec![(-1, 0), (sqrt_output, 0)]);
    graph.outputs.push((sqrt_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
