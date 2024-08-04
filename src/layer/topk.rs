use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use copy_constraint::zero_padding_partition;
use ndarray::{arr1, ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;

// Helper function to get the indices of the topk tensor
fn get_topk_indices(sorted_data_shape: Vec<usize>, k: usize) -> ArrayD<Option<IxDyn>> {
  let mut output_shape = sorted_data_shape.clone();
  let output_ndim = output_shape.len();
  output_shape[output_ndim - 1] = k;
  let padded_output_shape: Vec<_> = output_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();

  let topk_idx = ArrayD::from_shape_fn(output_shape.as_slice(), |index| Some(index.clone()));
  let padded_topk_idx = util::pad_to_pow_of_two(&topk_idx, &None);
  assert!(padded_topk_idx.shape() == padded_output_shape.as_slice());

  padded_topk_idx
}

// TopKLayer is a layer that returns the top k elements of the input tensor along a given axis
// The order of the elements is determined by the 'largest' attribute, the default '1' means descending
pub struct TopKLayer;
impl Layer for TopKLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let mut descending = true;
    let k = constants[1].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect::<Vec<_>>()[0] as usize;
    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;
    let largest = attributes.iter().filter(|x| x.name == "largest").next().unwrap().i as usize;
    if largest != 1 {
      descending = false;
    }

    let range = graph.addBB(Box::new(RangeConstBasicBlock {
      start: 0,
      limit: util::next_pow(input_shapes[0][axis] as u32) as i32,
      delta: 1,
    }));
    let data_len = input_shapes[0][axis];
    let sort = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SortBasicBlock {
        descending: descending,
        len: data_len,
      }),
      N: 1,
    }));
    let r: Vec<_> = if descending {
      (0..-onnx::CQ_RANGE_LOWER).map(Fr::from).collect()
    } else {
      (onnx::CQ_RANGE_LOWER + 1..1).map(Fr::from).collect()
    };
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { setup: arr1(&r) }),
      N: 1,
    }));

    if axis != input_shapes[0].len() - 1 {
      todo!("TopkLayer: axis != - 1; not implemented yet");
    }

    let range_output = graph.addNode(range, vec![]);
    let sort_output = graph.addNode(sort, vec![(-1, 0), (range_output, 0)]);
    let sorted_data_shape = input_shapes[0].clone();
    let padded_sorted_data_shape: Vec<_> = sorted_data_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let permutation = get_topk_indices(sorted_data_shape.clone(), k);
    let padding_partitions = zero_padding_partition(&permutation);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation.clone(),
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partitions,
    }));
    let topk_data_output = graph.addNode(cc, vec![(sort_output, 0)]);
    let topk_indices_output = graph.addNode(cc, vec![(sort_output, 1)]);

    let permutation_for_check = get_topk_indices(sorted_data_shape, data_len - 1);
    let padding_partitions = zero_padding_partition(&permutation_for_check);
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_for_check.clone(),
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partitions,
    }));
    let diff_data_output = graph.addNode(cc1, vec![(sort_output, 2)]);
    let _ = graph.addNode(range_check, vec![(diff_data_output, 0)]);
    let mut output_shape = permutation.shape().to_vec();

    let output_ndim = output_shape.len();
    output_shape[output_ndim - 1] = k;
    graph.outputs.push((topk_data_output, 0));
    graph.outputs.push((topk_indices_output, 0));
    (graph, vec![output_shape; 2])
  }
}
