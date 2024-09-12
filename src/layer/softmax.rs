use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct SoftmaxLayer;
impl Layer for SoftmaxLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let max = graph.addBB(Box::new(MaxBasicBlock {}));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let exp = graph.addBB(Box::new(ExpBasicBlock {
      input_SF: *onnx::SF_LOG,
      output_SF: *onnx::SF_LOG,
    }));
    let exp_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ExpBasicBlock {
            input_SF: *onnx::SF_LOG,
            output_SF: *onnx::SF_LOG,
          }),
          -(*onnx::CQ_RANGE as i32) + 1,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock {}),
      N: 1,
    }));
    let reciprocal = graph.addBB(Box::new(ReciprocalBasicBlock {
      input_SF: *onnx::SF_LOG,
      output_SF: *onnx::SF_LOG,
    }));
    let rec_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ReciprocalBasicBlock {
            input_SF: *onnx::SF_LOG,
            output_SF: *onnx::SF_LOG,
          }),
          0,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
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

    // The proving idea is as follows
    // 1. m = max(X): 
    //    We first compute the maximum value of the input array. 
    // 2. X - m: 
    //    We subtract the maximum value from each element of the input array.
    // 3. e^(X - m) * SF: 
    //    We compute the exponential of each element of the input array. 
    //    And we use "exp_check" to ensure that the output is within the CQ range.
    // 4. SUM(e^(X - m)) * SF: 
    //    We compute the sum of the exponential of each element of the input array.
    // 5. SF / SUM(e^(X - m)):
    //    We compute the reciprocal of the sum of the exponential of each element of the input array.
    //    And we use "rec_check" to ensure that the output is within the CQ range.
    // 6. [e^(X - m) * SF] * [SF / SUM(e^(X - m))]:
    //    We multiply the output from step 3 and step 5.
    // 7. [e^(X - m) * SF] * [SF / SUM(e^(X - m))] --> [e^(X - m) / SUM(e^(X - m))] * SF
    //    Change the scale factor of the output to the original scale factor.
    let max_output = graph.addNode(max, vec![(-1, 0)]);
    let sub_output = graph.addNode(sub, vec![(-1, 0), (max_output, 0)]);
    let exp_output = graph.addNode(exp, vec![(sub_output, 0)]);
    let _ = graph.addNode(exp_check, vec![(sub_output, 0), (exp_output, 0)]);
    let sum_output = graph.addNode(sum, vec![(exp_output, 0)]);
    let rec_output = graph.addNode(reciprocal, vec![(sum_output, 0)]);
    let _ = graph.addNode(rec_check, vec![(sum_output, 0), (rec_output, 0)]);
    let mul_output = graph.addNode(mul, vec![(exp_output, 0), (rec_output, 0)]);
    let output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (output, 0)]);
    graph.outputs.push((output, 0));

    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
