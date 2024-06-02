use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::indices;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::Slice;
use tract_onnx::pb::AttributeProto;

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

pub fn out_hw(dims: &Vec<usize>, strides: &Vec<usize>, ch_dims: &Vec<usize>, padding: &Vec<[usize; 2]>) -> Vec<usize> {
  dims.iter().enumerate().map(|(i, x)| (*x - ch_dims[i] + padding[i][0] + padding[i][1]) / strides[i] + 1).collect()
}

pub fn splat_input(input_shape: &Vec<usize>, strides: &Vec<usize>, pads: &Vec<usize>, ci: usize, ch_dims: &Vec<usize>) -> Vec<Vec<Option<IxDyn>>> {
  let dims = input_shape[2..].to_vec();
  let mut padding = vec![[0, 0], [0, 0]];
  for i in 0..dims.len() {
    padding.push([pads[i], pads[dims.len() + i]]);
  }

  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| x.into_dyn()).collect()).unwrap();

  let inp_pad = pad(&inp, &padding, &IxDyn::zeros(input_shape.len()));

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

pub fn splat_weights(weights_shape: &Vec<usize>) -> Vec<Vec<Option<IxDyn>>> {
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
fn splat_pad(input: &Vec<Vec<Option<IxDyn>>>) -> ArrayD<Option<IxDyn>> {
  let outp_size = input.len();
  let conv_size = input[0].len();
  let flattened_inp: Vec<_> = input.into_iter().flat_map(|x| x.iter().map(|y| y.clone())).collect();
  let flattened_inp = ArrayD::from_shape_vec(IxDyn(&vec![outp_size, conv_size]), flattened_inp).unwrap();
  let (m, n) = (flattened_inp.shape()[0].next_power_of_two(), flattened_inp.shape()[1].next_power_of_two());
  let (ph, pw) = ((0, m - flattened_inp.shape()[0]), (0, n - flattened_inp.shape()[1]));
  let padding = vec![[ph.0, ph.1], [pw.0, pw.1]];
  pad(&flattened_inp, &padding, &None)
}

pub struct ConvLayer;
impl Layer for ConvLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let weight_shape = input_shapes[1];
    let dims = input_shapes[0][2..].to_vec();
    let ch_dims = weight_shape[2..].to_vec();

    let strides: Vec<_> = match attributes.iter().filter(|x| x.name == "strides").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; dims.len()],
    };
    let pads: Vec<_> = match attributes.iter().filter(|x| x.name == "pads").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 2 * dims.len()],
    };

    // Splat input
    let permutation = splat_input(&input_shapes[0], &strides, &pads, weight_shape[1], &ch_dims);
    let permutation_padded = splat_pad(&permutation);
    let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_padded,
      input_dim: IxDyn(&input_shape_padded),
    }));

    // TODO: change to CQLin and commit splatted weights
    let weights_splatted = splat_weights(&weight_shape);
    let weights_padded = splat_pad(&weights_splatted);
    let weight_shape_padded: Vec<_> = weight_shape.iter().map(|i| i.next_power_of_two()).collect();
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: weights_padded,
      input_dim: IxDyn(&weight_shape_padded),
    }));
    let matmul = graph.addBB(Box::new(MatMulBasicBlock {}));

    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: onnx::SF_LOG * 2,
      output_SF: onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: Some((-(1 << 10), 1 << 11)),
      }),
      N: 1,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Reshape matmul into output shape
    let reshape = ArrayD::from_shape_fn(IxDyn(&[permutation.len(), weights_splatted.len()]), |i| Some(i));
    let mut padding = vec![];
    for i in 0..dims.len() {
      padding.push([pads[i], pads[dims.len() + i]]);
    }
    let mut out_dims = out_hw(&dims, &strides, &ch_dims, &padding);
    let mut output_shape = vec![1, weight_shape[0]];
    output_shape.append(&mut out_dims);

    let reshape_output = reshape.into_shape(IxDyn(&output_shape)).unwrap();

    let mut padding = vec![];
    for i in 0..output_shape.len() {
      padding.push([0, output_shape[i].next_power_of_two() - output_shape[i]]);
    }
    let reshape_padded = pad(&reshape_output, &padding, &None);
    let cc2 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: reshape_padded,
      input_dim: IxDyn(&[permutation.len().next_power_of_two(), weights_splatted.len().next_power_of_two()]),
    }));

    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    let cc1_output = graph.addNode(cc1, vec![(-2, 0)]);
    let cqlin_output = graph.addNode(matmul, vec![(cc_output, 0), (cc1_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(change_SF_output, 0)]);
    let add_output = graph.addNode(add, vec![(change_SF_output, 0), (-3, 0)]);
    let cc2_output = graph.addNode(cc2, vec![(add_output, 0)]);
    graph.outputs.push((cc2_output, 0));
    (graph, vec![output_shape])
  }
}
