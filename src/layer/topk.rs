use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

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
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let mut descending = true;
    let k = constants[1].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect::<Vec<_>>()[0] as usize;
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
    let one_to_one = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(OneToOneBasicBlock {}),
      N: 1,
    }));
    let ordered = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(OrderedBasicBlock {}),
      N: 1,
    }));
    let r: Vec<_> = if descending {
      (0..-*onnx::CQ_RANGE_LOWER).map(Fr::from).collect()
    } else {
      (*onnx::CQ_RANGE_LOWER + 1..1).map(Fr::from).collect()
    };
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { setup: arr1(&r) }),
      N: 1,
    }));

    let sorted_data_shape = input_shapes[0].clone();
    let padded_sorted_data_shape: Vec<_> = sorted_data_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();

    let permutation = get_topk_indices(sorted_data_shape.clone(), k);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation.clone(),
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let permutation_for_ordered_check = get_topk_indices(sorted_data_shape, data_len - 1);
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_for_ordered_check.clone(),
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    if axis != input_shapes[0].len() - 1 {
      todo!("TopkLayer: axis != - 1; not implemented yet");
    }

    // The overview of the proving:
    // 1. Create indices, a range from 0 to the length of the input tensor along the axis
    // 2. Sort the input tensor and the corresponding indices (step 1.) along the axis
    // 3. Check if the sorted tensor and sorted indices are 1-to-1 mapped from the input tensor and the indices
    // 4. Check if the sorted tensor is ordered by checking if the difference between consecutive elements
    //    is non-negative (if descending) or non-positive (if ascending)
    // 5. Copy the first k elements of the sorted tensor and the sorted indices
    let range_output = graph.addNode(range, vec![]);
    let sort_output = graph.addNode(sort, vec![(-1, 0), (range_output, 0)]);
    let _ = graph.addNode(one_to_one, vec![(-1, 0), (range_output, 0), (sort_output, 0), (sort_output, 1)]);

    let diff_data_output = graph.addNode(ordered, vec![(sort_output, 0)]);
    let diff_data_output = graph.addNode(cc1, vec![(diff_data_output, 0)]);
    let _ = graph.addNode(range_check, vec![(diff_data_output, 0)]);

    let topk_data_output = graph.addNode(cc, vec![(sort_output, 0)]);
    let topk_indices_output = graph.addNode(cc, vec![(sort_output, 1)]);

    let mut output_shape = input_shapes[0].clone();

    let output_ndim = output_shape.len();
    output_shape[output_ndim - 1] = k;
    graph.outputs.push((topk_data_output, 0));
    graph.outputs.push((topk_indices_output, 0));
    (graph, vec![output_shape; 2], vec![input_types[0], DatumType::I64])
  }
}

// ArgMaxLayer is a layer that returns the top 1 index of the input tensor along a given axis
pub struct ArgMaxLayer;
impl Layer for ArgMaxLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let descending = true;
    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;

    let range = graph.addBB(Box::new(RangeConstBasicBlock {
      start: 0,
      limit: util::next_pow(input_shapes[0][axis] as u32) as i32,
      delta: 1,
    }));
    let data_len = input_shapes[0][axis];
    let sort = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SortBasicBlock { descending, len: data_len }),
      N: 1,
    }));
    let one_to_one = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(OneToOneBasicBlock {}),
      N: 1,
    }));
    let ordered = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(OrderedBasicBlock {}),
      N: 1,
    }));
    let r: Vec<_> = (0..-*onnx::CQ_RANGE_LOWER).map(Fr::from).collect();
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { setup: arr1(&r) }),
      N: 1,
    }));

    let sorted_data_shape = input_shapes[0].clone();
    let padded_sorted_data_shape: Vec<_> = sorted_data_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();

    let permutation = get_topk_indices(sorted_data_shape.clone(), 1);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let permutation_for_ordered_check = get_topk_indices(sorted_data_shape, data_len - 1);
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_for_ordered_check,
      input_dim: IxDyn(&padded_sorted_data_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    if axis != input_shapes[0].len() - 1 {
      todo!("ArgMaxLayer: axis != - 1; not implemented yet");
    }

    // The proving is similar to the TopKLayer, but we only need to copy the top 1 element
    let range_output = graph.addNode(range, vec![]);
    let sort_output = graph.addNode(sort, vec![(-1, 0), (range_output, 0)]);
    let _ = graph.addNode(one_to_one, vec![(-1, 0), (range_output, 0), (sort_output, 0), (sort_output, 1)]);

    let diff_data_output = graph.addNode(ordered, vec![(sort_output, 0)]);
    let diff_data_output = graph.addNode(cc1, vec![(diff_data_output, 0)]);
    let _ = graph.addNode(range_check, vec![(diff_data_output, 0)]);

    let top1_indices_output = graph.addNode(cc, vec![(sort_output, 1)]);

    let mut output_shape = input_shapes[0].clone();

    let output_ndim = output_shape.len();
    output_shape[output_ndim - 1] = 1;
    graph.outputs.push((top1_indices_output, 0));
    (graph, vec![output_shape], vec![DatumType::I64])
  }
}
