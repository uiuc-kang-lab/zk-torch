use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use copy_constraint::zero_padding_partition;
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{Array1, ArrayD};
use tract_onnx::pb::AttributeProto;

pub fn out_hw(dims: &Vec<usize>, strides: &Vec<usize>, ch_dims: &Vec<usize>, padding: &Vec<[usize; 2]>) -> Vec<usize> {
  dims.iter().enumerate().map(|(i, x)| (*x - ch_dims[i] + padding[i][0] + padding[i][1]) / strides[i] + 1).collect()
}

fn splat_input(input_shape: &Vec<usize>, strides: &Vec<usize>, pads: &Vec<usize>, ci: usize, ch_dims: &Vec<usize>) -> Vec<Vec<Option<IxDyn>>> {
  let dims = input_shape[2..].to_vec();
  let mut padding = vec![[0, 0], [0, 0]];
  for i in 0..dims.len() {
    padding.push([pads[i], pads[dims.len() + i]]);
  }

  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| x.into_dyn()).collect()).unwrap();

  let inp_pad = util::pad(&inp, &padding, &IxDyn::zeros(input_shape.len()));

  let out_dims = out_hw(&dims, &strides, &ch_dims, &padding[2..].to_vec());

  let mut inp_cells = vec![];
  let mut input_row_idx = 0;

  // (out_dims product x inp_channels * ch_dims product)
  for batch in 0..inp.shape()[0] {
    for out_idx in indices(out_dims.clone()) {
      inp_cells.push(vec![]);
      for ck in 0..ci {
        for ch_idx in indices(IxDyn(&ch_dims)) {
          let mut idx = vec![batch, ck];
          idx.append(&mut (0..dims.len()).map(|i| out_idx[i] * strides[i] + ch_idx[i]).collect());
          inp_cells[input_row_idx].push(Some(inp_pad[IxDyn(&idx)].clone()));
        }
      }
      input_row_idx += 1;
    }
  }
  inp_cells
}

fn splat_weights(weights_shape: &Vec<usize>) -> Vec<Vec<Option<IxDyn>>> {
  let mut weights_cells = vec![];
  let mut weight_row_idx = 0;

  let out_channels = weights_shape[0];
  for chan_out in 0..out_channels {
    weights_cells.push(vec![]);
    for ck in 0..weights_shape[1] {
      for ch_idx in indices(IxDyn(&weights_shape[2..])) {
        let mut idx = vec![chan_out, ck];
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

// Returns the permutation for CopyConstraintBasicBlock for a reshape operation given the unpadded input and output shapes
pub fn reshape_permutation(input_shape: &Vec<usize>, output_shape: &Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let reshape = ArrayD::from_shape_fn(IxDyn(&input_shape), |i| Some(i));

  let reshape_output = reshape.into_shape(IxDyn(&output_shape)).unwrap();

  let mut padding = vec![];
  for i in 0..output_shape.len() {
    padding.push([0, output_shape[i].next_power_of_two() - output_shape[i]]);
  }
  let reshape_padded = util::pad(&reshape_output, &padding, &None);
  reshape_padded
}

pub struct ConvLayer;
impl Layer for ConvLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
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
    let permutation = splat_input(&input_shapes[0], &strides, &pads, weight_shape[1], &ch_dims);
    let permutation_padded = splat_pad(&permutation);
    let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
    let padding_partitions = zero_padding_partition(&permutation_padded);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_padded,
      input_dim: IxDyn(&input_shape_padded),
      padding_partitions,
    }));

    // TODO: change to CQLin and commit splatted weights
    let weights_splatted = splat_weights(&weight_shape);
    let weights_padded = splat_pad(&weights_splatted);
    let weight_shape_padded: Vec<_> = weight_shape.iter().map(|i| i.next_power_of_two()).collect();
    let padding_partitions = zero_padding_partition(&weights_padded);
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: weights_padded,
      input_dim: IxDyn(&weight_shape_padded),
      padding_partitions,
    }));
    let matmul = graph.addBB(Box::new(MatMulBasicBlock {}));

    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: onnx::SF_LOG * 2,
      output_SF: onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: Array1::from_iter(onnx::CQ_RANGE_LOWER..-onnx::CQ_RANGE_LOWER).map(|x| Fr::from(*x)),
      }),
      N: 1,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Reshape matmul into output shape
    let mut padding = vec![];
    for i in 0..dims.len() {
      padding.push([pads[i], pads[dims.len() + i]]);
    }
    let mut out_dims = out_hw(&dims, &strides, &ch_dims, &padding);
    let mut output_shape = vec![1, weight_shape[0]];
    output_shape.append(&mut out_dims);
    let reshape_permutation = reshape_permutation(&vec![permutation.len(), weights_splatted.len()], &output_shape);
    let padding_partitions = zero_padding_partition(&reshape_permutation);
    let cc2 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: reshape_permutation,
      input_dim: IxDyn(&[permutation.len().next_power_of_two(), weights_splatted.len().next_power_of_two()]),
      padding_partitions,
    }));

    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    let cc1_output = graph.addNode(cc1, vec![(-2, 0)]);
    let cqlin_output = graph.addNode(matmul, vec![(cc_output, 0), (cc1_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(change_SF_output, 0)]);

    // Add bias if it exists
    let add_output = {
      if input_shapes.len() > 2 {
        graph.addNode(add, vec![(change_SF_output, 0), (-3, 0)])
      } else {
        change_SF_output
      }
    };
    let cc2_output = graph.addNode(cc2, vec![(add_output, 0)]);
    graph.outputs.push((cc2_output, 0));
    (graph, vec![output_shape])
  }
}
