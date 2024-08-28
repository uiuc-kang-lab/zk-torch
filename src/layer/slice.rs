use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

fn combinations<T: Clone>(vecs: &Vec<Vec<T>>) -> Vec<Vec<T>> {
  // Recursive function to generate combinations
  fn combine<T: Clone>(vecs: &[Vec<T>], current: Vec<T>, result: &mut Vec<Vec<T>>) {
    if vecs.is_empty() {
      result.push(current);
    } else {
      for item in &vecs[0] {
        let mut new_current = current.clone();
        new_current.push(item.clone());
        combine(&vecs[1..], new_current, result);
      }
    }
  }

  let mut result = Vec::new();
  combine(&vecs, Vec::new(), &mut result);
  result
}

fn get_slice(
  input_dim: &Vec<usize>,
  starts: &mut Vec<usize>,
  ends: &mut Vec<usize>,
  axes: &mut Vec<usize>,
  steps: &mut Vec<usize>,
) -> (ArrayD<Option<IxDyn>>, Vec<usize>, Vec<usize>) {
  let rank = input_dim.len();
  let mut result_idx = vec![vec![]; rank];
  let mut real_output_shape = vec![0; rank];
  let mut real_ends = ends.clone();

  let input_shape_pad: Vec<_> = input_dim.iter().map(|&x| util::next_pow(x as u32) as usize).collect();

  if starts.len() < rank {
    for i in 0..rank {
      if axes.contains(&i) {
        continue;
      }
      starts.insert(i, 0);
      ends.insert(i, input_shape_pad[i]);
      real_ends.insert(i, input_dim[i]);
      axes.insert(i, i);
      steps.insert(i, 1);
    }
  }

  for (i, &axis) in axes.iter().enumerate() {
    let step = steps[i];
    let mut start = starts[i];
    let end = ends[i];
    let mut real_end = real_ends[i];
    if end > input_shape_pad[i] {
      real_end = input_dim[i];
    }
    while start < real_end {
      result_idx[axis].push(start);
      real_output_shape[axis] += 1;
      start += step;
    }
  }
  let combination_result = combinations(&result_idx);
  let f = combination_result.iter().map(|v| Some(IxDyn(v))).collect();
  let result = ArrayD::from_shape_vec(real_output_shape.clone(), f).unwrap();
  let result = util::pad_to_pow_of_two(&result, &None);
  (result, real_output_shape, input_shape_pad)
}

// https://onnx.ai/onnx/operators/onnx__Slice.html
pub struct SliceLayer;
impl Layer for SliceLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let starts: Vec<_> = constants[1].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect();
    let ends: Vec<_> = constants[2].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect();
    // steps and axes might be optional
    let axes: Vec<_> = match constants.get(3) {
      Some(x) => x.unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect(),
      None => (0..input_shapes[0].len()).map(|x| x as i32).collect(),
    };
    let mut steps: Vec<_> = match constants.get(4) {
      Some(x) => x.unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).collect(),
      None => vec![1; starts.len()],
    };
    let mut axes: Vec<_> = axes
      .iter()
      .map(|&x| {
        if x < 0 {
          (input_shapes[0].len() as i32 + x) as usize
        } else {
          x as usize
        }
      })
      .collect();
    let mut starts: Vec<_> = starts
      .iter()
      .enumerate()
      .map(|(i, &x)| {
        if x < 0 {
          (input_shapes[0][axes[i]] as i32 + x) as usize
        } else {
          x as usize
        }
      })
      .collect();
    let mut ends: Vec<_> = ends
      .iter()
      .enumerate()
      .map(|(i, &x)| {
        if x < 0 {
          (input_shapes[0][axes[i]] as i32 + x) as usize
        } else {
          x as usize
        }
      })
      .collect();

    let (permutation, output_shape, input_shape_pad) = get_slice(&input_shapes[0], &mut starts, &mut ends, &mut axes, &mut steps);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&input_shape_pad),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let slice_output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((slice_output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
