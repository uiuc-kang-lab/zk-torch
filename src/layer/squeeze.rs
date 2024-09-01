use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util::{self, get_reshape_indices};
use ark_bn254::Fr;
use ndarray::{ArrayD, Axis, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Squeeze the input tensor by removing all dimensions of size 1.
// If axes is provided, remove the dimensions specified by axes.
// Otherwise, remove all dimensions of size 1.
// If the last dimension is squeezed, we need to permute the tensor before reshaping because the last dimension affects the commitment.
pub struct SqueezeLayer;
impl Layer for SqueezeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let axes_result = attributes.iter().filter(|x| x.name == "axes").next();
    let mut axes: Vec<_>;
    if let Some(x) = axes_result {
      // axes is provided
      axes = x.ints.iter().map(|x| *x as i64).collect();
    } else {
      // axes is not provided
      axes = match constants.get(1) {
        Some(x) => x.unwrap().0.iter().map(|x| util::fr_to_int(*x) as i64).collect(),
        _ => input_shapes[0].iter().enumerate().filter(|(_, x)| **x == 1).map(|(i, _)| i as i64).collect(),
      };
    }

    // map negative axes to positive
    axes = axes.iter().map(|&x| if x < 0 { input_shapes[0].len() as i64 + x } else { x }).collect();

    let startShape = input_shapes[0];
    assert!(axes.iter().all(|&x| startShape[x as usize] == 1));
    let endShape: Vec<_> = startShape.iter().enumerate().filter(|(i, _)| !axes.contains(&(*i as i64))).map(|(_, x)| *x).collect();

    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock {
        shape: endShape.clone().iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else {
      let startShape_padded: Vec<_> = startShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
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

#[derive(Debug)]
pub struct UnsqueezeBasicBlock;
impl BasicBlock for UnsqueezeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // unsqueeze the input tensor
    let r = inputs[0].clone();
    let r = r.insert_axis(Axis(0));
    vec![r]
  }
}

// Unsqueeze the input tensor by adding a dimension of size 1 at the specified axis.
// If the last dimension is unsqueezed, we need to permute the tensor before reshaping because the last dimension affects the commitment.
// Otherwise, when the last dimension is not unsqueezed or an arr0 is unsqueezed (special case), we can directly reshape it.
pub struct UnsqueezeLayer;
impl Layer for UnsqueezeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let axis: isize = match attributes.iter().filter(|x| x.name == "axes").next() {
      Some(v) => v.ints[0] as isize,
      None => util::fr_to_int(constants[1].unwrap().0[0]) as isize,
    };
    let axis = if axis < 0 { input_shapes[0].len() as isize + axis + 1 } else { axis };

    let startShape = input_shapes[0];
    let endShape: Vec<_> = (0..startShape.len() + 1)
      .map(|x| {
        if x == axis as usize {
          1
        } else {
          if x > axis as usize {
            startShape[x - 1]
          } else {
            startShape[x]
          }
        }
      })
      .collect();

    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock {
        shape: endShape.clone().iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else if startShape.last() > endShape.last() {
      let n = endShape.len();
      let mut a = endShape[n - 2];
      assert!(*startShape.last().unwrap() == a);
      let mut intermediateShape = endShape[..n - 2].to_vec();
      intermediateShape.push(1);
      intermediateShape.push(*startShape.last().unwrap());
      intermediateShape.iter_mut().for_each(|x| *x = util::next_pow(*x as u32) as usize);
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: intermediateShape }));
      a = util::next_pow(a as u32) as usize;
      let permutation = ((0..a).map(|x| x).collect(), vec![0]);
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let intermediate = graph.addNode(reshape, vec![(-1, 0)]);
      let output = graph.addNode(permute, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    } else {
      // special case (startShape.last() < endShape.last()): [] --> [1]
      let unsqueeze = graph.addBB(Box::new(UnsqueezeBasicBlock {}));
      let unsqueeze_output = graph.addNode(unsqueeze, vec![(-1, 0)]);
      graph.outputs.push((unsqueeze_output, 0));
    }

    (graph, vec![endShape], vec![input_types[0]])
  }
}
