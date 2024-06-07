use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;

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

fn get_slice(input_dim: &Vec<usize>, starts: &Vec<usize>, ends: &Vec<usize>, axes: &Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let rank = input_dim.len();
  let steps = vec![1; starts.len()];
  let mut result_idx = vec![vec![]; rank];
  let mut result_shape = vec![0; rank];

  for (i, &axis) in axes.iter().enumerate() {
    let step = steps[i];
    let mut start = starts[i];
    let end = ends[i];
    while start < end {
      result_idx[axis].push(start);
      result_shape[axis] += 1;
      start += step;
    }
  }
  let combination_result = combinations(&result_idx);
  let f = combination_result.iter().map(|v| Some(IxDyn(v))).collect();
  let result = ArrayD::from_shape_vec(result_shape, f).unwrap();
  result
}

pub struct SliceLayer;
impl Layer for SliceLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    // TODO: handle negative indices and optional inputs
    let starts: Vec<_> = constants[1].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).collect();
    let ends: Vec<_> = constants[2].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).collect();
    let axes = constants[3].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).collect();

    let permutation = get_slice(&input_shapes[0], &starts, &ends, &axes);
    let output_shape = permutation.shape().to_vec();

    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation,
      input_dim: IxDyn(&input_shapes[0]),
    }));
    let slice_output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((slice_output, 0));

    (graph, vec![output_shape])
  }
}
