use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::iterable::Iterable;
use ndarray::Dimension;
use ndarray::{ArrayD, Axis, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// array: the N-dimensional array
// n_minus_1_index: the index of the N-1 dimension
fn get_sub_array<T>(array: ArrayD<T>, n_minus_1_index: &[usize]) -> ArrayD<T>
where
  T: Clone,
{
  let mut sub_array = array.clone();
  for &index in n_minus_1_index.iter() {
    let s = sub_array.view();
    sub_array = s.index_axis(Axis(0), index).to_owned();
  }
  sub_array
}

fn get_gathernd_masks(input_shape: &[usize], indices: &ArrayD<usize>, batch_dims: usize) -> (ArrayD<Option<IxDyn>>, Vec<usize>) {
  assert!(indices.shape()[indices.ndim() - 1] <= input_shape.len() - batch_dims);
  // ref: https://docs.openvino.ai/2022.3/openvino_docs_ops_movement_GatherND_8.html
  let output_shape: Vec<usize> = if indices.shape()[indices.ndim() - 1] == input_shape.len() - batch_dims {
    // indice shape but exclude the last dimension
    indices.shape().iter().take(indices.ndim() - 1).cloned().collect()
  } else {
    // indices.shape[:batch_dims] + list(indices.shape)[batch_dims:-1] + list(data.shape)[batch_dims + indices.shape[-1]:].
    let mut output_shape = vec![];
    output_shape.extend_from_slice(&indices.shape()[..indices.ndim() - 1]);
    output_shape.extend_from_slice(&input_shape[batch_dims + indices.shape()[indices.ndim() - 1]..]);
    output_shape
  };

  // permutation[i_0, ..., i_{K-2},:,...,:] = [indices[i_0, ..., i_{K-2}],:,...,:]
  let permutation = ArrayD::from_shape_fn(output_shape.clone(), |idx| {
    let mut v = vec![];
    // select the partial index from 0..indices.len() - 1
    let mut partial_idx = vec![];
    for i in 0..indices.ndim() - 1 {
      partial_idx.push(idx[i]);
    }
    let sub_array = get_sub_array(indices.clone(), &partial_idx);
    v.extend(sub_array.as_slice().unwrap());
    for i in indices.ndim() - 1..idx.ndim() {
      v.push(idx[i]);
    }
    Some(IxDyn(&v))
  });

  let padded_permutation = util::pad_to_pow_of_two(&permutation, &None);
  (padded_permutation, output_shape)
}

// reference (v13): https://onnx.ai/onnx/operators/onnx__GatherND.html
pub struct GatherNDLayer;
impl Layer for GatherNDLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let indices = if constants[1].is_none() {
      // we cannot handle non-constant indices because we need to know the shape of the indices to compile graph in zk-torch
      panic!("GatherNDLayer: indices must be a constant");
    } else {
      constants[1].unwrap().0.map(|x| util::fr_to_int(*x) as usize)
    };

    // attributes may contain batch_dims, but we only support batch_dims = 0 for now
    let batch_dims: usize = if attributes.iter().find(|x| x.name == "batch_dims").is_none() {
      0
    } else {
      let b = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as usize;
      if b != 0 {
        panic!("GatherNDLayer: only support the case where batch_dims = 0");
      } else {
        b
      }
    };

    let data_shape = input_shapes[0].clone();
    let padded_data_shape: Vec<_> = data_shape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();

    let (permutation, output_shape) = get_gathernd_masks(&data_shape, &indices, batch_dims);

    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&padded_data_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
