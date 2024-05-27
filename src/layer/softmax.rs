use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct SoftmaxLayer;
impl Layer for SoftmaxLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let max = graph.addBB(Box::new(MaxBasicBlock {}));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let exp = graph.addBB(Box::new(ExpBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let exp_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ExpBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          -(onnx::CQ_RANGE as i32),
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock {}),
      N: 1,
    }));
    let log = graph.addBB(Box::new(LogBasicBlock {
      input_SF: onnx::SF_LOG,
      output_SF: onnx::SF_LOG,
    }));
    let log_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(LogBasicBlock {
            input_SF: onnx::SF_LOG,
            output_SF: onnx::SF_LOG,
          }),
          0,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    let max_output = graph.addNode(max, vec![(-1, 0)]);
    let sub_output = graph.addNode(sub, vec![(-1, 0), (max_output, 0)]);
    let exp_output = graph.addNode(exp, vec![(sub_output, 0)]);
    let _ = graph.addNode(exp_check, vec![(sub_output, 0), (exp_output, 0)]);
    let sum_output = graph.addNode(sum, vec![(exp_output, 0)]);
    let log_output = graph.addNode(log, vec![(sum_output, 0)]);
    let _ = graph.addNode(log_check, vec![(sum_output, 0), (log_output, 0)]);
    let add_output = graph.addNode(add, vec![(log_output, 0), (max_output, 0)]);
    let sub_output_2 = graph.addNode(sub, vec![(-1, 0), (add_output, 0)]);
    let exp_output_2 = graph.addNode(exp, vec![(sub_output_2, 0)]);
    let _ = graph.addNode(exp_check, vec![(sub_output_2, 0), (exp_output_2, 0)]);
    graph.outputs.push((exp_output_2, 0));

    (graph, vec![input_shapes[0].clone()])
  }
}
