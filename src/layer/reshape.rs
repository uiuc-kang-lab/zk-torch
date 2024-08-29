use crate::basic_block::*;
use crate::graph::*;
use crate::layer::{squeeze::UnsqueezeBasicBlock, Layer};
use crate::util::{self, get_reshape_indices};
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct ReshapeLayer;
impl Layer for ReshapeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let startShape = input_shapes[0];
    let mut endShape: Vec<_> = constants[1]
      .unwrap()
      .0
      .as_slice()
      .unwrap()
      .iter()
      .enumerate()
      .map(|(i, x)| {
        if i < input_shapes[1][0] {
          // If a shape dimension is 0, then we replace the value with the corresponding input dimension
          if *x == Fr::zero() {
            input_shapes[0][i] as i32
          } else {
            util::fr_to_int(*x)
          }
        } else {
          0
        }
      })
      .filter(|x| *x != 0)
      .collect();
    if let Some(i) = endShape.iter().position(|&x| x == -1) {
      let a = input_shapes[0].iter().fold(1, |x, &y| x * y) as i32;
      let b = endShape.iter().fold(-1, |x, &y| x * y);
      endShape[i] = a / b;
    }
    let endShape: Vec<_> = endShape.iter().map(|&x| x as usize).filter(|x| *x != 0).collect();
    let endShape_padded: Vec<_> = endShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
    let startShape_padded: Vec<_> = startShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
    // check if the product of startShape_padded is equal to the product of endShape_padded
    let equal = startShape_padded.iter().fold(1, |x, &y| x * y) == endShape_padded.iter().fold(1, |x, &y| x * y);

    if equal && (startShape.last() == endShape.last()) {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock {
        shape: endShape_padded.clone(),
      }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else if startShape.len() == 0 {
      // special case: arr0 --> [1,1,...]
      let unsq = graph.addBB(Box::new(UnsqueezeBasicBlock {}));
      let mut unsq_output = graph.addNode(unsq, vec![(-1, 0)]);
      for _ in 0..endShape.len() - 1 {
        unsq_output = graph.addNode(unsq, vec![(unsq_output, 0)]);
      }
      graph.outputs.push((unsq_output, 0));
    } else {
      let permutation = get_reshape_indices(startShape.clone(), endShape.clone());
      let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: permutation.clone(),
        input_dim: IxDyn(&startShape_padded),
        padding_partition: copy_constraint::PaddingEnum::Zero,
      }));
      let output = graph.addNode(cc, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape], vec![input_types[0]])
  }
}
