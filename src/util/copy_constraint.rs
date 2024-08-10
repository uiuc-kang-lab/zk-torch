/*
 * Copy constraint utilities:
 * The functions are used for constructing the permutation and
 * padding_partitions fields in the CopyConstraintBasicBlock.
 */

use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{ArrayD, Axis, Dimension, IxDyn};
use std::collections::HashMap;

// Returns the padding partition where the non-zero padding value consists of all pad indices such that the last-axis subview containing it contains non-pad elements, and the zero padding value consists of all pad indices part of a last-axis subview containing only pad elements.
// If val is 0, then these will instead be combined.
pub fn max_padding_partitions(permutation: &ArrayD<Option<IxDyn>>, val: Fr) -> HashMap<Fr, Vec<IxDyn>> {
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
  let mut partitions = HashMap::new();
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
