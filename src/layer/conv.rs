use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
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

// Splats weights into shape (out_channels x inp_channels * kernel_dims product) such that each row corresponds to flattened kernels for each input channel
fn splat_weights(weights_shape: &Vec<usize>, is_transpose: bool) -> Vec<Vec<Option<IxDyn>>> {
  let mut weights_cells = vec![];
  let mut weight_row_idx = 0;

  // Input and output channel positions are swapped between Conv and ConvTranspose
  let out_channels = if is_transpose { weights_shape[1] } else { weights_shape[0] };
  let in_channels = if is_transpose { weights_shape[0] } else { weights_shape[1] };
  // (out_channels x inp_channels * kernel_dims product)
  for chan_out in 0..out_channels {
    weights_cells.push(vec![]);
    for ck in 0..in_channels {
      for ch_idx in indices(IxDyn(&weights_shape[2..])) {
        let mut idx = if is_transpose { vec![ck, chan_out] } else { vec![chan_out, ck] };
        idx.append(&mut ch_idx.as_array_view().to_vec());
        weights_cells[weight_row_idx].push(Some(IxDyn(&idx)));
      }
    }
    weight_row_idx += 1;
  }
  weights_cells
}

// Adds padding to the nearest power of two to splatted inputs/weights
pub fn splat_pad(input: &Vec<Vec<Option<IxDyn>>>) -> ArrayD<Option<IxDyn>> {
  let outp_size = input.len();
  let conv_size = input[0].len();
  let flattened_inp: Vec<_> = input.into_iter().flat_map(|x| x.iter().map(|y| y.clone())).collect();
  let flattened_inp = ArrayD::from_shape_vec(IxDyn(&vec![outp_size, conv_size]), flattened_inp).unwrap();
  util::pad_to_pow_of_two(&flattened_inp, &None)
}

// The overview of the proving:
// 1. Splat inputs into (out_dims product x inp_channels * kernel_dims product)
// 2. Splat weights into (out_channels x inp_channels * kernel_dims product)
// 3. Perform matmul between inputs and weights with the resulting shape (out_dims product x out_channels). Each element corresponds to an element of the final output
// 4. Scale down and add bias if it exists
// 5. Reshape into the final output shape
macro_rules! create_conv_layer {
  ($layer_name:ident, $is_transpose:expr) => {
    pub struct $layer_name;

    impl Layer for $layer_name {
      fn graph(
        input_shapes: &Vec<&Vec<usize>>,
        input_types: &Vec<DatumType>,
        _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
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

        // Splat input
        let ci = if $is_transpose { weight_shape[0] } else { weight_shape[1] };
        let permutation = splat_input(&input_shapes[0], &strides, &pads, ci, &ch_dims, $is_transpose);
        let permutation_padded = splat_pad(&permutation);
        let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
        let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
          permutation: permutation_padded.clone(),
          input_dim: IxDyn(&input_shape_padded),
          padding_partition: copy_constraint::PaddingEnum::Zero,
        }));

        // TODO: change to CQLin and commit splatted weights
        // let weights_splatted = splat_weights(&weight_shape, $is_transpose);
        // let weights_padded = splat_pad(&weights_splatted);
        // let weight_shape_padded: Vec<_> = weight_shape.iter().map(|i| i.next_power_of_two()).collect();
        // let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
        //   permutation: weights_padded,
        //   input_dim: IxDyn(&weight_shape_padded),
        //   padding_partition: copy_constraint::PaddingEnum::Zero,
        // }));
        // let matmul = graph.addBB(Box::new(MatMulBasicBlock {}));

        // let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
        //   input_SF: *onnx::SF_LOG * 2,
        //   output_SF: *onnx::SF_LOG,
        // }));
        // let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
        //   basic_block: Box::new(CQBasicBlock {
        //     setup: Array1::from_iter(*onnx::CQ_RANGE_LOWER..-*onnx::CQ_RANGE_LOWER).map(|x| Fr::from(*x)),
        //   }),
        //   N: 1,
        // }));

        // // Add bias
        // let add = graph.addBB(Box::new(RepeaterBasicBlock {
        //   basic_block: Box::new(AddBasicBlock {}),
        //   N: 1,
        // }));

        // // Reshape matmul into output shape
        // let mut padding = vec![];
        // let pads = if $is_transpose {
        //   conv_transpose_pads(&pads, &ch_dims)
        // } else {
        //   pads.clone()
        // };
        // for i in 0..dims.len() {
        //   padding.push([pads[i], pads[dims.len() + i]]);
        // }
        // let mut out_dims = out_hw(&dims, &strides, &ch_dims, &padding, $is_transpose);
        // let cout = if $is_transpose { weight_shape[1] } else { weight_shape[0] };
        // let mut output_shape = vec![1, cout];
        // output_shape.append(&mut out_dims);
        // let reshape_permutation = util::get_reshape_indices(vec![permutation.len(), weights_splatted.len()], output_shape.clone());
        // let cc2 = graph.addBB(Box::new(CopyConstraintBasicBlock {
        //   permutation: reshape_permutation,
        //   input_dim: IxDyn(&[permutation.len().next_power_of_two(), weights_splatted.len().next_power_of_two()]),
        //   padding_partition: copy_constraint::PaddingEnum::Zero,
        // }));

        let cc_output = graph.addNode(cc, vec![(-1, 0)]);
        // let cc1_output = graph.addNode(cc1, vec![(-2, 0)]);
        // let cqlin_output = graph.addNode(matmul, vec![(cc_output, 0), (cc1_output, 0)]);
        // let change_SF_output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
        // let _ = graph.addNode(change_SF_check, vec![(change_SF_output, 0)]);

        // // Add bias if it exists
        // let add_output = {
        //   if input_shapes.len() > 2 {
        //     graph.addNode(add, vec![(change_SF_output, 0), (-3, 0)])
        //   } else {
        //     change_SF_output
        //   }
        // };
        // let cc2_output = graph.addNode(cc2, vec![(add_output, 0)]);
        graph.outputs.push((cc_output, 0));
        (graph, vec![permutation_padded.shape().to_vec()], vec![input_types[0]])
      }
    }
  };
}

create_conv_layer!(ConvLayer, false);
create_conv_layer!(ConvTransposeLayer, true);
