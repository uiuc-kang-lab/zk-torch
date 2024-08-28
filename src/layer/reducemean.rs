use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// ReduceMeanLayer is a layer that returns the mean of the input tensor along one or two given axis/axes
// More than two axes is not supported for now
pub struct ReduceMeanLayer;
impl Layer for ReduceMeanLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let axes: Vec<_> = match attributes.iter().filter(|x| x.name == "axes").next() {
      Some(x) => x.ints.iter().map(|x| *x).collect(),
      None => vec![(input_shapes[0].len() - 1) as i64],
    };

    let axes: Vec<_> = axes
      .iter()
      .map(|&x| {
        if x < 0 {
          (input_shapes[0].len() as i64 + x) as usize
        } else {
          x as usize
        }
      })
      .collect();

    // Only support reducing along one or two axis
    assert!(axes.len() == 1 || axes.len() == 2);
    // reducing along the last axis
    assert!(axes.iter().any(|&x| x == input_shapes[0].len() - 1));

    let n = input_shapes[0].len();
    let mut a = input_shapes[0][n - 1];
    a = util::next_pow(a as u32) as usize;
    let permutation = (vec![0], (0..a).collect());
    // PermuteBasicBlock is used for permute the last two dimensions for the case of reducing along two axes
    // (we need it because our mean computation is done along the last dimension)
    let permute = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
      N: 2,
    }));

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
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let sum_output = graph.addNode(sum, vec![(-1, 0)]);
    let div_output = graph.addNode(div, vec![(sum_output, 0)]);
    let _ = graph.addNode(div_check, vec![(sum_output, 0), (div_output, 0)]);

    if axes.len() == 2 {
      let permute_output = graph.addNode(permute, vec![(div_output, 0)]);
      let sum_output1 = graph.addNode(sum, vec![(permute_output, 0)]);
      let div_output1 = graph.addNode(div, vec![(sum_output1, 0)]);
      let _ = graph.addNode(div_check, vec![(sum_output1, 0), (div_output1, 0)]);
      graph.outputs.push((div_output1, 0));
      let mut outputShape = input_shapes[0].clone();
      outputShape[input_shapes[0].len() - 1] = 1;
      outputShape[input_shapes[0].len() - 2] = 1;
      (graph, vec![outputShape], vec![input_types[0]])
    } else if axes.len() == 1 {
      graph.outputs.push((div_output, 0));
      let mut outputShape = input_shapes[0].clone();
      outputShape[input_shapes[0].len() - 1] = 1;
      (graph, vec![outputShape], vec![input_types[0]])
    } else {
      panic!("Only support reducing along one or two axis");
    }
  }
}
