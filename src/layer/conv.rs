use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use ark_bn254::Fr;
use ark_ff::Zero;
use ndarray::indices;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::Slice;
use tract_onnx::pb::AttributeProto;

pub fn pad<G: Clone>(input: &ArrayD<G>, padding: &Vec<[usize; 2]>, pad_val: &G) -> ArrayD<G> {
  let tmp = input.iter().collect();
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

pub fn out_hw(h: usize, w: usize, si: usize, sj: usize, ch: usize, cw: usize, padding: &Vec<[usize; 2]>) -> (usize, usize) {
  (
    (h - ch + padding[2][0] + padding[2][1]) / si + 1,
    (w - cw + 2 * padding[3][0] + padding[3][1]) / sj + 1,
  )
}

pub fn splat_input(input_shape: &Vec<usize>, stride: &Vec<usize>, padding: &Vec<usize>, ci: usize, ch: usize, cw: usize) -> ArrayD<IxDyn> {
  let h: usize = input_shape[2];
  let w: usize = input_shape[3];

  let (si, sj) = (stride[0], stride[1]);

  assert_eq!(input_shape.len(), 4);
  let padding = vec![[0, 0], [0, 0], [padding[0], padding[2]], [padding[1], padding[3]]];

  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| x.into_dyn()).collect()).unwrap();

  // for pad: we can do option type, NONE would just be 0s
  let inp_pad = pad(&inp, &padding, &IxDyn(&[0, 0, 0, 0]));

  let (oh, ow) = out_hw(h, w, si, sj, ch, cw, &padding);

  let mut inp_cells = vec![];
  let mut input_row_idx = 0;

  // (O_H * O_W x inp_channels * C_H * C_W)
  for batch in 0..inp.shape()[0] {
    for i in 0..oh {
      for j in 0..ow {
        inp_cells.push(vec![]);
        for ck in 0..ci {
          for ci in 0..ch {
            for cj in 0..cw {
              let idx_i = i * si + ci;
              let idx_j = j * sj + cj;
              inp_cells[input_row_idx].push(inp_pad[[batch, ck, idx_i, idx_j]].clone());
            }
          }
        }
        input_row_idx += 1;
      }
    }
  }
  let batch_size = input_shape[0];
  let conv_size = inp_cells[0].len();
  let flattened_inp: Vec<_> = inp_cells.into_iter().flat_map(|x| x.into_iter()).collect();
  let flattened_inp = ArrayD::from_shape_vec(IxDyn(&vec![batch_size * oh * ow, conv_size]), flattened_inp).unwrap();
  let (m, n) = (flattened_inp.shape()[0].next_power_of_two(), flattened_inp.shape()[1].next_power_of_two());
  let (ph, pw) = ((0, m - flattened_inp.shape()[0]), (0, n - flattened_inp.shape()[1]));
  let padding = vec![[ph.0, ph.1], [pw.0, pw.1]];
  let splat_inp_pad = pad(&flattened_inp, &padding, &IxDyn(&[0, 0, 0, 0]));
  splat_inp_pad
}

pub fn splat_weights(weights: &ArrayD<Fr>) -> ArrayD<Fr> {
  // B, H, W, C
  // println!("oh, ow: {}, {}", oh, ow);

  let mut weights_cells = vec![];
  let mut weight_row_idx = 0;

  // (output_channels x inp_channels * C_H * C_W)
  for chan_out in 0..weights.shape()[0] {
    weights_cells.push(vec![]);
    for ck in 0..weights.shape()[1] {
      for ci in 0..weights.shape()[2] {
        for cj in 0..weights.shape()[3] {
          weights_cells[weight_row_idx].push(weights[[chan_out, ck, ci, cj]].clone());
        }
      }
    }
    weight_row_idx += 1;
  }
  let out_channels = weights.shape()[0];
  let flattened_weights: Vec<_> = weights_cells.into_iter().flat_map(|x| x.into_iter()).collect();
  let conv_size = flattened_weights.len() / out_channels;
  let weights_array = ArrayD::from_shape_vec(IxDyn(&vec![out_channels, conv_size]), flattened_weights).unwrap().t().to_owned();
  let (m, n) = (weights_array.shape()[0].next_power_of_two(), weights_array.shape()[1].next_power_of_two());
  let (ph, pw) = ((0, m - weights_array.shape()[0]), (0, n - weights_array.shape()[1]));
  let padding = vec![[ph.0, ph.1], [pw.0, pw.1]];
  let splat_weights_pad = pad(&weights_array, &padding, &Fr::zero());
  splat_weights_pad
}
