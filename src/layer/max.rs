use crate::basic_block::*;
use crate::graph::*;
use crate::layer::conv::reshape_permutation;
use crate::layer::Layer;
use crate::util::pad_to_pow_of_two;
use ark_bn254::Fr;
use ark_std::Zero;
use copy_constraint::zero_padding_partition;
use ndarray::{concatenate, indices, ArrayD, Axis, Dim, Dimension, IxDyn};
use std::collections::BTreeMap;
use tract_onnx::pb::AttributeProto;

// Returns the splat needed to pass into MaxProofBasicBlock. This produces a (product of input dims X 2) permutation where the first column corresponds to the input elements and the second column contains None for the constant values
fn splat_input(input_shape: &Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| Some(x.into_dyn())).collect()).unwrap();
  let inp = inp.into_shape(IxDyn(&[input_shape.iter().product(), 1])).unwrap();
  let inp_pad = pad_to_pow_of_two(&inp, &None);
  let none_col = ArrayD::from_elem(inp_pad.shape(), None);
  concatenate(Axis(1), &[inp_pad.view(), none_col.view()]).unwrap()
}

// Returns the padding partition where the non-zero padding value consists of all pad indices such that the last-axis subview containing it contains non-pad elements, and the zero padding value consists of all pad indices part of a last-axis subview containing only pad elements.
// If val is 0, then these will instead be combined.
fn max_padding_partitions(permutation: &ArrayD<Option<IxDyn>>, val: Fr) -> BTreeMap<Fr, Vec<IxDyn>> {
  let mut zero_indices = vec![];
  let mut nonzero_indices = vec![];
  for (i, subview) in permutation.axis_iter(Axis(permutation.ndim() - 1)).enumerate() {
    if subview.iter().all(|x| x.is_none()) {
      for (idx, _) in subview.indexed_iter() {
        let mut full_idx = idx.as_array_view().to_vec();
        full_idx.push(i);
        zero_indices.push(IxDyn(&full_idx));
      }
    } else {
      for (idx, val) in subview.indexed_iter() {
        if val.is_none() {
          let mut full_idx = idx.as_array_view().to_vec();
          full_idx.push(i);
          nonzero_indices.push(IxDyn(&full_idx));
        }
      }
    }
  }
  let mut partitions = BTreeMap::new();
  if val == Fr::zero() {
    zero_indices.append(&mut nonzero_indices);
  } else {
    if nonzero_indices.len() > 0 {
      partitions.insert(val, nonzero_indices);
    }
  }
  if zero_indices.len() > 0 {
    partitions.insert(Fr::zero(), zero_indices);
  }
  partitions
}

pub struct MaxLayer;
impl Layer for MaxLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    // For now we only support the case when there are two inputs and the second input is a constant of a single element
    if input_shapes.len() == 2 && input_shapes[1].len() == 1 && constants[1].is_some() {
      let constant = constants[1].unwrap().first().unwrap();
      let permutation = splat_input(&input_shapes[0]);
      let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
      let padding_partitions = max_padding_partitions(&permutation, *constant);
      let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation,
        input_dim: IxDyn(&input_shape_padded),
        padding_partitions,
      }));
      let max = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MaxProofBasicBlock {}),
        N: 1,
      }));
      let reshape_shape = &vec![input_shapes[0].iter().product(), 1];
      let reshape_permutation = reshape_permutation(&reshape_shape, &input_shapes[0]);
      let padding_partitions = zero_padding_partition(&reshape_permutation);
      let reshape_shape_pad: Vec<_> = reshape_shape.iter().map(|i| i.next_power_of_two()).collect();
      let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: reshape_permutation,
        input_dim: IxDyn(&reshape_shape_pad),
        padding_partitions,
      }));

      let cc_output = graph.addNode(cc, vec![(-1, 0)]);
      let max_output = graph.addNode(max, vec![(cc_output, 0)]);
      let cc1_output = graph.addNode(cc1, vec![(max_output, 0)]);
      graph.outputs.push((cc1_output, 0));
    } else {
      panic!("MaxLayer only supports having two inputs where the second input is a constant")
    }
    (graph, vec![input_shapes[0].clone()])
  }
}
