/*
 * Shape utilities:
 * The functions are used for shape-related operations, such as
 * slicing and padding arrays.
 */
use ark_bn254::Fr;
use ndarray::{ArrayD, Axis, IxDyn, Slice, SliceInfo};

// slice the arr with the given indices. But this function is not used in the codebase currently.
#[allow(dead_code)]
pub fn slice_nd_array(arr: ArrayD<Fr>, indices: &[usize]) -> ArrayD<Fr> {
  // Create slices from the indices
  let slices: Vec<_> = indices.iter().map(|&i| (0..i).into()).collect();

  // Convert slices into a SliceInfo instance
  let slice_info = unsafe { SliceInfo::<_, IxDyn, IxDyn>::new(slices).unwrap() };

  // Slice the array
  arr.slice_move(slice_info)
}

// Pads each dimension of input by the corresponding amount in padding on both ends.
pub fn pad<G: Clone>(input: &ArrayD<G>, padding: &Vec<[usize; 2]>, pad_val: &G) -> ArrayD<G> {
  let tmp = input.into_iter().collect();
  let input = ArrayD::from_shape_vec(input.raw_dim(), tmp).unwrap();
  assert_eq!(input.ndim(), padding.len());
  let mut padded_shape = input.raw_dim();
  for (ax, (&ax_len, &[pad_lo, pad_hi])) in input.shape().iter().zip(padding).enumerate() {
    padded_shape[ax] = ax_len + pad_lo + pad_hi;
  }

  let mut padded = ArrayD::from_elem(padded_shape, pad_val);
  let padded_dim = padded.raw_dim();
  {
    // Select portion of padded array that needs to be copied from the
    // original array.
    let mut orig_portion = padded.view_mut();
    for (ax, &[pad_lo, pad_hi]) in padding.iter().enumerate() {
      orig_portion.slice_axis_inplace(Axis(ax), Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)));
    }
    // Copy the data from the original array.
    orig_portion.assign(&input);
  }

  let dim = padded.raw_dim();
  let tmp = padded.into_iter().map(|x| x.clone()).collect();
  let padded = ArrayD::from_shape_vec(dim, tmp).unwrap();

  padded
}

pub fn pad_to_pow_of_two<G: Clone>(input: &ArrayD<G>, pad_val: &G) -> ArrayD<G> {
  let padding: Vec<_> = input.shape().iter().map(|x| [0, x.next_power_of_two() - x]).collect();
  pad(&input, &padding, &pad_val)
}

pub fn broadcastDims(dims: &Vec<&Vec<usize>>, N: usize) -> Vec<usize> {
  let len = dims.iter().map(|x| x.len()).max().unwrap();
  (0..len - N)
    .map(|i| dims.iter().map(|dim| if dim.len() >= len - i { dim[i + dim.len() - len] } else { 1 }).max().unwrap())
    .collect()
}
