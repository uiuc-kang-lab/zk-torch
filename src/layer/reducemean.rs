use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct ReduceMeanLayer;
impl Layer for ReduceMeanLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock {}),
      N: 1,
    }));
    let div = graph.addBB(Box::new(DivConstBasicBlock {
      c: input_shapes[0][input_shapes[0].len() - 1] as f32,
    }));
    let div_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(DivConstBasicBlock {
            c: input_shapes[0][input_shapes[0].len() - 1] as f32,
          }),
          onnx::CQ_RANGE_LOWER,
          onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sum_output = graph.addNode(sum, vec![(-1, 0)]);
    let div_output = graph.addNode(div, vec![(sum_output, 0)]);
    let _ = graph.addNode(div_check, vec![(sum_output, 0), (div_output, 0)]);
    graph.outputs.push((div_output, 0));

    let mut outputShape = input_shapes[0].clone();
    outputShape[input_shapes[0].len() - 1] = 1;
    (graph, vec![outputShape])
  }
}
