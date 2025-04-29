use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{Array1, ArrayD};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Returns output dimensions given the actual padding amount
pub fn out_hw(dims: &Vec<usize>, strides: &Vec<usize>, ch_dims: &Vec<usize>, padding: &Vec<[usize; 2]>, is_transpose: bool) -> Vec<usize> {
  if is_transpose {
    dims
      .iter()
      .enumerate()
      .map(|(i, x)| (strides[i] * (*x - 1) + ch_dims[i] - (ch_dims[i] - 1 - padding[i][0]) - (ch_dims[i] - 1 - padding[i][1])))
      .collect()
  } else {
    dims.iter().enumerate().map(|(i, x)| (*x - ch_dims[i] + padding[i][0] + padding[i][1]) / strides[i] + 1).collect()
  }
}

// Output the actual padding amount added to the input (kernel_size - 1 - pads)
fn conv_transpose_pads(pads: &Vec<usize>, ch_dims: &Vec<usize>) -> Vec<usize> {
  pads.iter().enumerate().map(|(i, x)| ch_dims[i % ch_dims.len()] - 1 - *x).collect()
}

// Splats inputs into shape (out_dims product x inp_channels * kernel_dims product) such that each row corresponds to a flattened kernel row in the splatted weights
// In the case of ci = 1 (input channels), the row should correspond to the flattened bottom plane in this visualization: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
fn splat_input(
  input_shape: &Vec<usize>,
  strides: &Vec<usize>,
  pads: &Vec<usize>,
  ci: usize,
  kernel_dims: &Vec<usize>,
  is_transpose: bool,
) -> Vec<Vec<Option<IxDyn>>> {
  let dims = input_shape[2..].to_vec();
  let pads = if is_transpose {
    conv_transpose_pads(pads, kernel_dims)
  } else {
    pads.clone()
  };
  let mut padding = vec![[0, 0], [0, 0]];
  for i in 0..dims.len() {
    padding.push([pads[i], pads[dims.len() + i]]);
  }

  let out_dims = out_hw(&dims, &strides, &kernel_dims, &padding[2..].to_vec(), is_transpose);
  let inp = if is_transpose {
    // if stride > 1, expand input by stride amount
    let mut pre_pad_dims: Vec<_> = out_dims.iter().zip(pads.iter()).map(|(x, y)| x - y).collect();
    let mut inp_shape = input_shape[..2].to_vec();
    inp_shape.append(&mut pre_pad_dims);
    ArrayD::from_shape_fn(inp_shape, |inp_idx| {
      let ch_idx = inp_idx.as_array_view().to_vec()[2..].to_vec();
      if ch_idx.iter().zip(strides).all(|(&idx, stride)| idx % stride == 0) {
        let mut perm_idx = inp_idx.as_array_view().to_vec()[..2].to_vec();
        perm_idx.append(&mut ch_idx.iter().zip(strides).map(|(&idx, stride)| idx / stride).collect());
        Some(IxDyn(&perm_idx))
      } else {
        None
      }
    })
  } else {
    let inp_shape = Dim(IxDyn(input_shape));
    ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| Some(x.into_dyn())).collect()).unwrap()
  };
  let inp_pad = util::pad(&inp, &padding, &None);

  let mut inp_cells = vec![];
  let mut input_row_idx = 0;

  // (out_dims product x inp_channels * kernel_dims product)
  for batch in 0..inp.shape()[0] {
    for out_idx in indices(out_dims.clone()) {
      inp_cells.push(vec![]);
      for ck in 0..ci {
        for ch_idx in indices(IxDyn(&kernel_dims)) {
          let mut idx = vec![batch, ck];
          if is_transpose {
            idx.append(&mut (0..dims.len()).map(|i| out_idx[i] + ch_idx[i]).collect());
          } else {
            idx.append(&mut (0..dims.len()).map(|i| out_idx[i] * strides[i] + ch_idx[i]).collect());
          }
          inp_cells[input_row_idx].push(inp_pad[IxDyn(&idx)].clone());
        }
      }
      input_row_idx += 1;
    }
  }
  inp_cells
}

// Splats weights into shape (inp_channels * kernel_dims product x out_channels) such that each row corresponds to flattened kernels for each input channel
fn splat_weights(weights: &ArrayD<Fr>, is_transpose: bool) -> Vec<Vec<Fr>> {
  let mut weights_cells = vec![];
  let mut weight_row_idx = 0;

  // Input and output channel positions are swapped between Conv and ConvTranspose
  let out_channels = if is_transpose { weights.shape()[1] } else { weights.shape()[0] };
  let in_channels = if is_transpose { weights.shape()[0] } else { weights.shape()[1] };
  // (inp_channels * kernel_dims product x out_channels)
  for ck in 0..in_channels {
    for ch_idx in indices(IxDyn(&weights.shape()[2..])) {
      weights_cells.push(vec![]);
      for chan_out in 0..out_channels {
        let mut idx = if is_transpose { vec![ck, chan_out] } else { vec![chan_out, ck] };
        idx.append(&mut ch_idx.as_array_view().to_vec());
        weights_cells[weight_row_idx].push(weights[IxDyn(&idx)]);
      }
      weight_row_idx += 1;
    }
  }
  weights_cells
}

// Adds padding to the nearest power of two to splatted inputs/weights
pub fn splat_pad<G: Clone>(input: &Vec<Vec<G>>, pad_val: &G) -> ArrayD<G> {
  let outp_size = input.len();
  let conv_size = input[0].len();
  let flattened_inp: Vec<_> = input.into_iter().flat_map(|x| x.iter().map(|y| y.clone())).collect();
  let flattened_inp = ArrayD::from_shape_vec(IxDyn(&vec![outp_size, conv_size]), flattened_inp).unwrap();
  util::pad_to_pow_of_two(&flattened_inp, pad_val)
}

// The overview of the proving:
// 1. Splat inputs into (out_dims product x inp_channels * kernel_dims product)
// 1a. If 1x1 kernel, instead we transpose and permute the input so that inp_channels is the last dimension
// 2. Perform matmul between inputs and weights with the resulting shape (out_dims product x out_channels). Each element corresponds to an element of the final output
// 2a. If 1x1 kernel, the last dimension is out_channels and the rest are unchanged
// 3. Scale down and add bias if it exists
// 4. Reshape into the final output shape
macro_rules! create_conv_layer {
  ($layer_name:ident, $is_transpose:expr) => {
    pub struct $layer_name;

    impl Layer for $layer_name {
      fn graph(
        input_shapes: &Vec<&Vec<usize>>,
        input_types: &Vec<DatumType>,
        constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
        attributes: &Vec<&AttributeProto>,
      ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
        let mut graph = Graph::new();
        let weight_shape = input_shapes[1];
        let dims = input_shapes[0][2..].to_vec();
        let ch_dims = weight_shape[2..].to_vec();

        let strides = match attributes.iter().filter(|x| x.name == "strides").next() {
          Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
          None => vec![1; dims.len()],
        };
        let pads = match attributes.iter().filter(|x| x.name == "pads").next() {
          Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
          None => vec![0; 2 * dims.len()],
        };
        let _dilations = match attributes.iter().filter(|x| x.name == "dilations").next() {
          Some(v) => v
            .ints
            .iter()
            .map(|x| {
              if *x != 1 {
                panic!("dilations != 1 not supported");
              }
              *x as usize
            })
            .collect(),
          None => vec![1; dims.len() - 2],
        };

        // Add bias
        let add = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(AddBasicBlock {}),
          N: 1,
        }));

        // Matmul
        let weights_splatted = splat_weights(constants[1].unwrap().0, $is_transpose);
        let weights_padded = splat_pad(&weights_splatted, &Fr::zero());
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQLinBasicBlock { setup: weights_padded }),
          N: 2,
        }));

        // Scale down
        let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
        let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
          input_SF: sf_log * 2,
          output_SF: sf_log,
        }));
        let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQ2BasicBlock {
            n: weights_splatted[0].len().next_power_of_two(),
            setup: Some((
              Box::new(ChangeSFBasicBlock {
                input_SF: sf_log * 2,
                output_SF: sf_log,
              }),
              *onnx::CQ_RANGE_LOWER,
              *onnx::CQ_RANGE,
            )),
          }),
          N: 1,
        }));

        // Splat input with transpose+permute with 1x1 optimization or with copy constraint otherwise
        let ci = if $is_transpose { weight_shape[0] } else { weight_shape[1] };
        let permutation = splat_input(&input_shapes[0], &strides, &pads, ci, &ch_dims, $is_transpose);
        let n = input_shapes[0].len();
        let kernel_1x1_opt = pads.iter().all(|&x| x == 0) && strides.iter().all(|&x| x == 1) && ch_dims[0] == 1;
        let cc = if kernel_1x1_opt {
          let mut perm = vec![0];
          perm.append(&mut (2..n - 1).collect());
          perm.append(&mut vec![1, n - 1]);
          graph.addBB(Box::new(TransposeBasicBlock { perm }))
        } else {
          let permutation_padded = splat_pad(&permutation, &None);
          let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
          graph.addBB(Box::new(CopyConstraintBasicBlock {
            permutation: permutation_padded,
            input_dim: IxDyn(&input_shape_padded),
            padding_partition: copy_constraint::PaddingEnum::Zero,
          }))
        };

        // Only used if 1x1 kernel
        let a = input_shapes[0][1].next_power_of_two();
        let b = input_shapes[0][n - 1].next_power_of_two();
        let transpose1 = ((0..b).map(|x| x * a).collect(), (0..a).collect());
        let permute1 = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(PermuteBasicBlock {
            permutation: transpose1,
            n: a,
            m: b,
          }),
          N: 2,
        }));
        let transpose2 = ((0..a).map(|x| x * b).collect(), (0..b).collect());
        let permute2 = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(PermuteBasicBlock {
            permutation: transpose2,
            n: b,
            m: a,
          }),
          N: 2,
        }));

        // Reshape matmul into output shape with transpose+permute with 1x1 optimization or with copy constraint otherwise
        let mut padding = vec![];
        let pads = if $is_transpose {
          conv_transpose_pads(&pads, &ch_dims)
        } else {
          pads.clone()
        };
        for i in 0..dims.len() {
          padding.push([pads[i], pads[dims.len() + i]]);
        }
        let mut out_dims = out_hw(&dims, &strides, &ch_dims, &padding, $is_transpose);
        let cout = if $is_transpose { weight_shape[1] } else { weight_shape[0] };
        let mut output_shape = vec![1, cout];
        output_shape.append(&mut out_dims);

        let cc2 = if kernel_1x1_opt {
          let mut perm = vec![0, n - 2];
          let n = input_shapes[0].len();
          perm.append(&mut (1..n - 2).collect());
          perm.append(&mut vec![n - 1]);
          graph.addBB(Box::new(TransposeBasicBlock { perm }))
        } else {
          let output_shape_padded = output_shape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
          let reshape_permutation = util::get_reshape_indices(vec![permutation.len(), weights_splatted[0].len()], output_shape_padded);
          graph.addBB(Box::new(CopyConstraintBasicBlock {
            permutation: reshape_permutation,
            input_dim: IxDyn(&[permutation.len().next_power_of_two(), weights_splatted[0].len().next_power_of_two()]),
            padding_partition: copy_constraint::PaddingEnum::Zero,
          }))
        };

        // Splat input with transpose+permute with 1x1 optimization or with copy constraint otherwise
        let cc_output = if kernel_1x1_opt {
          let trans_output = graph.addNode(cc, vec![(-1, 0)]);
          graph.addNode(permute1, vec![(trans_output, 0)])
        } else {
          graph.addNode(cc, vec![(-1, 0)])
        };

        // Matmul
        let cqlin_output = graph.addNode(cqlin, vec![(cc_output, 0)]);
        let change_SF_output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
        let _ = graph.addNode(change_SF_check, vec![(cqlin_output, 0), (change_SF_output, 0)]);

        // Add bias if it exists
        let add_output = {
          if input_shapes.len() > 2 {
            graph.addNode(add, vec![(change_SF_output, 0), (-3, 0)])
          } else {
            change_SF_output
          }
        };

        // Reshape matmul into output shape with transpose+permute with 1x1 optimization or with copy constraint otherwise
        let cc2_output = if kernel_1x1_opt {
          let trans_output = graph.addNode(permute2, vec![(add_output, 0)]);
          graph.addNode(cc2, vec![(trans_output, 0)])
        } else {
          graph.addNode(cc2, vec![(add_output, 0)])
        };
        graph.outputs.push((cc2_output, 0));
        (graph, vec![output_shape], vec![input_types[0]])
      }
    }
  };
}

create_conv_layer!(ConvLayer, false);
create_conv_layer!(ConvTransposeLayer, true);
