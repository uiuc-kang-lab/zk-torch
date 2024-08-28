/*
 * Copy constraint utilities:
 * The functions are used for constructing the permutation and
 * padding_partitions fields in the CopyConstraintBasicBlock.
 */
use crate::util::pad_to_pow_of_two;
use ndarray::{ArrayD, IxDyn};

// Helper function to get the indices of the reshaped tensor
// Note that the input_shape and output_shape are non-padded
pub fn get_reshape_indices(input_shape: Vec<usize>, output_shape: Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let indices = ArrayD::from_shape_fn(input_shape.as_slice(), |index| Some(index.clone()));
  let output_indices = indices.view().into_shape(&output_shape[..]).unwrap().to_owned();

  let padded_indices = pad_to_pow_of_two(&output_indices, &None);
  padded_indices
}
