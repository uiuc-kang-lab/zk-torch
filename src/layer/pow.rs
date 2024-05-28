use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct PowLayer;
impl Layer for PowLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    assert!(constants[1].unwrap().first().unwrap() == &Fr::from(2 * crate::onnx::SF as u32));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: crate::onnx::SF_LOG * 2,
      output_SF: crate::onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: crate::onnx::SF_LOG * 2,
            output_SF: crate::onnx::SF_LOG,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let mul_output = graph.addNode(mul, vec![(-1, 0), (-1, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    graph.outputs.push((change_SF_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
