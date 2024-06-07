use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct CeilLayer;
impl Layer for CeilLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let relu = graph.addBB(Box::new(CeilBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let relu_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(CeilBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let relu_output = graph.addNode(relu, vec![(-1, 0)]);
    let _ = graph.addNode(relu_check, vec![(-1, 0), (relu_output, 0)]);
    graph.outputs.push((relu_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
