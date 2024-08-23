use crate::basic_block::*;
use crate::graph::*;
use crate::layer::{squeeze::UnsqueezeBasicBlock, Layer};
use crate::util::{self, get_reshape_indices};
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct ReshapeLayer;
impl Layer for ReshapeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let startShape = input_shapes[0];
    let mut endShape: Vec<_> = constants[1].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).filter(|x| *x != 0).collect();
    if let Some(i) = endShape.iter().position(|&x| x == -1) {
      let a = input_shapes[0].iter().fold(1, |x, &y| x * y) as i32;
      let b = endShape.iter().fold(-1, |x, &y| x * y);
      endShape[i] = a / b;
    }
    let endShape: Vec<_> = endShape.iter().map(|&x| x as usize).filter(|x| *x != 0).collect();

    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: endShape.clone() }));
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
      let startShape_padded: Vec<_> = startShape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();
      let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: permutation.clone(),
        input_dim: IxDyn(&startShape_padded),
        padding_partition: copy_constraint::PaddingEnum::Zero,
      }));
      let output = graph.addNode(cc, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape])
  }
}
