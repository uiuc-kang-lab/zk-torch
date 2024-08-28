use crate::basic_block::*;
use crate::graph::*;
use crate::layer::conv::{out_hw, splat_pad};
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, indices, ArrayD, Dim, Dimension, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// This constructs the permutation for CopyConstraintBasicBlock to be inputted into MaxProofBasicBlock. The output is a (product of output dims of the pool operation * input channels X product of kernel_dims) permutation where the rows correspond to one max operation, and each row contains the set of arguments to max.
// ci is the number of input channels
fn splat_input(input_shape: &Vec<usize>, strides: &Vec<usize>, pads: &Vec<usize>, ci: usize, kernel_dims: &Vec<usize>) -> Vec<Vec<Option<IxDyn>>> {
  let dims = input_shape[2..].to_vec();
  let mut padding = vec![[0, 0], [0, 0]];
  for i in 0..dims.len() {
    padding.push([pads[i], pads[dims.len() + i]]);
  }

  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| x.into_dyn()).collect()).unwrap();

  let inp_pad = util::pad(&inp, &padding, &IxDyn::zeros(input_shape.len()));

  let out_dims = out_hw(&dims, &strides, &kernel_dims, &padding[2..].to_vec(), false);

  let mut inp_cells = vec![];
  let mut input_row_idx = 0;

  // (out_dims product * inp_channels x kernel_dims product)
  for batch in 0..inp.shape()[0] {
    for out_idx in indices(out_dims.clone()) {
      for ck in 0..ci {
        inp_cells.push(vec![]);
        for kernel_idx in indices(IxDyn(&kernel_dims)) {
          let mut idx = vec![batch, ck];
          idx.append(&mut (0..dims.len()).map(|i| out_idx[i] * strides[i] + kernel_idx[i]).collect());
          inp_cells[input_row_idx].push(Some(inp_pad[IxDyn(&idx)].clone()));
        }
        input_row_idx += 1;
      }
    }
  }
  inp_cells
}

pub struct MaxPoolLayer;
impl Layer for MaxPoolLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let dims = input_shapes[0][2..].to_vec();

    let kernel_shape: Vec<_> = attributes.iter().filter(|x| x.name == "kernel_shape").next().unwrap().ints.iter().map(|x| *x as usize).collect();

    let strides: Vec<_> = match attributes.iter().filter(|x| x.name == "strides").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; dims.len()],
    };
    let pads: Vec<_> = match attributes.iter().filter(|x| x.name == "pads").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 2 * dims.len()],
    };

    // Splat input
    let ch = input_shapes[0][1];
    let permutation = splat_input(&input_shapes[0], &strides, &pads, ch, &kernel_shape);
    let permutation_padded = splat_pad(&permutation);
    let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();

    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation_padded,
      input_dim: IxDyn(&input_shape_padded),
      padding_partition: copy_constraint::PaddingEnum::Max(Fr::from(*onnx::CQ_RANGE_LOWER)),
    }));

    // Prove max over each row
    let max = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MaxProofBasicBlock {
        cq_range_lower: *onnx::CQ_RANGE_LOWER,
      }),
      N: 1,
    }));

    // Reshape into output shape
    let mut padding = vec![[0, 0], [0, 0]];
    for i in 0..dims.len() {
      padding.push([pads[i], pads[dims.len() + i]]);
    }
    let mut output_shape = input_shapes[0][..2].to_vec();
    output_shape.append(&mut out_hw(&dims, &strides, &kernel_shape, &padding[2..].to_vec(), false));
    let reshape_inp_shape = vec![output_shape.iter().fold(1, |acc, &x| acc * x), 1];
    let reshape_permutation = util::get_reshape_indices(reshape_inp_shape.clone(), output_shape.clone());
    let reshape_inp_padded: Vec<_> = reshape_inp_shape.iter().map(|x| x.next_power_of_two()).collect();
    let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: reshape_permutation,
      input_dim: IxDyn(&reshape_inp_padded),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let r: Vec<_> = (0..-*onnx::CQ_RANGE_LOWER).map(Fr::from).collect();
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { setup: arr1(&r) }),
      N: 1,
    }));

    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    let max_output = graph.addNode(max, vec![(cc_output, 0)]);
    let cc1_output = graph.addNode(cc1, vec![(max_output, 0)]);
    let _ = graph.addNode(range_check, vec![(max_output, 1)]);
    graph.outputs.push((cc1_output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
