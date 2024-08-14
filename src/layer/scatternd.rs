use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use copy_constraint::zero_padding_partition;
use ndarray::{ArrayD, Dim, IxDyn};
use tract_onnx::pb::AttributeProto;

fn get_masks(input_shape: &[usize], indices: &ArrayD<Fr>) -> (ArrayD<Option<IxDyn>>, ArrayD<Option<IxDyn>>) {
  let mut preserve = ArrayD::from_shape_fn(input_shape, |index| Some(index));
  let mut update = ArrayD::from_shape_fn(input_shape, |_| None);
  let indices_usize = indices.map(|x| util::fr_to_int(*x) as usize);
  let indices_shape = indices.shape();
  let update_indices = &indices_shape[..indices_shape.len() - 1];

  let mut current_index = vec![];
  let mut all_indices = vec![];
  ndindex(update_indices, &mut current_index, &mut all_indices);

  for idx in all_indices {
    let update_index = indices_usize[Dim(idx.clone())];
    preserve[Dim(update_index)] = None;
    update[Dim(update_index)] = Some(Dim(idx));
  }

  (preserve, update)
}

fn ndindex(shape: &[usize], current_index: &mut Vec<usize>, all_indices: &mut Vec<Vec<usize>>) {
  if current_index.len() == shape.len() {
    all_indices.push(current_index.clone());
    return;
  }

  let dim = shape[current_index.len()];
  for i in 0..dim {
    current_index.push(i);
    ndindex(shape, current_index, all_indices);
    current_index.pop();
  }
}

// https://onnx.ai/onnx/operators/onnx__ScatterND.html
pub struct ScatterNDLayer;
impl Layer for ScatterNDLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    // TODO: handle cases when attribute is not none
    let indices = constants[1].unwrap();

    let (permutation_preserve, permutation_update) = get_masks(&input_shapes[0], &indices);
    let padding_preserve = zero_padding_partition(&permutation_preserve);
    let padding_update = zero_padding_partition(&permutation_update);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_preserve,
      input_dim: IxDyn(&input_shapes[0]),
      padding_partitions: padding_preserve,
    }));
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_update,
      input_dim: IxDyn(&input_shapes[1]),
      padding_partitions: padding_update,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    let data_to_preserve = graph.addNode(cc, vec![(-1, 0)]);
    let data_to_update = graph.addNode(cc1, vec![(-3, 0)]);
    let add_output = graph.addNode(add, vec![(data_to_preserve, 0), (data_to_update, 0)]);
    graph.outputs.push((add_output, 0));

    (graph, vec![input_shapes[0].clone()])
  }
}
