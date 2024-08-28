use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, Axis, Dim, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

fn get_masks(input_shape: &[usize], indices: &ArrayD<Fr>) -> (ArrayD<Option<IxDyn>>, ArrayD<Option<IxDyn>>) {
  let preserve = ArrayD::from_shape_fn(input_shape, |index| Some(index));
  let update = ArrayD::from_shape_fn(input_shape, |_| None);
  let mut preserve = util::pad_to_pow_of_two(&preserve, &None);
  let mut update = util::pad_to_pow_of_two(&update, &None);
  let indices_usize = indices.map(|x| util::fr_to_int(*x) as usize);
  let indices_shape = indices.shape();
  let update_indices = &indices_shape[..indices_shape.len() - 1];

  let mut current_index = vec![];
  let mut all_indices = vec![];
  ndindex(update_indices, &mut current_index, &mut all_indices);
  let update_indices = indices_usize.lanes(Axis(indices_usize.ndim() - 1));
  for (idx, update_idx) in all_indices.iter().zip(update_indices) {
    if update_idx.len() > input_shape.len() {
      // use only the first n elements of update_idx, where n is the length of input_shape
      let mut new_update_idx = update_idx.to_vec();
      new_update_idx.truncate(input_shape.len());
      let copy_update_idx = Dim(new_update_idx.to_vec());
      let copy_idx = Dim(idx.clone());
      preserve[copy_update_idx.clone()] = None;
      update[copy_update_idx] = Some(copy_idx.clone());
    } else if update_idx.len() < input_shape.len() {
      let input_shape_extra_dims = input_shape[input_shape.len() - update_idx.len()..].to_vec();
      let mut current_index = vec![];
      let mut extra_indices = vec![];
      ndindex(&input_shape_extra_dims, &mut current_index, &mut extra_indices);
      for extra_idx in extra_indices {
        // concat the extra indices to the update_idx
        let mut new_update_idx = update_idx.to_vec();
        new_update_idx.extend(extra_idx.clone());
        let copy_update_idx = Dim(new_update_idx.to_vec());
        // concat the extra indices to the idx
        let mut new_idx = idx.to_vec();
        new_idx.extend(extra_idx);
        let copy_idx = Dim(new_idx.to_vec());

        preserve[copy_update_idx.clone()] = None;
        update[copy_update_idx] = Some(copy_idx.clone());
      }
    } else {
      preserve[Dim(update_idx.to_vec())] = None;
      update[Dim(update_idx.to_vec())] = Some(Dim(idx.clone()));
    }
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
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    // TODO: handle cases when attribute is not none
    let indices = constants[1].unwrap().0;

    let (permutation_preserve, permutation_update) = get_masks(&input_shapes[0], &indices);
    let input_shape_0_padded: Vec<_> = input_shapes[0].iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let input_shape_2_padded: Vec<_> = input_shapes[2].iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_preserve,
      input_dim: IxDyn(&input_shape_0_padded),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_update,
      input_dim: IxDyn(&input_shape_2_padded),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    let data_to_preserve = graph.addNode(cc, vec![(-1, 0)]);
    let data_to_update = graph.addNode(cc1, vec![(-3, 0)]);
    let add_output = graph.addNode(add, vec![(data_to_preserve, 0), (data_to_update, 0)]);
    graph.outputs.push((add_output, 0));

    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
