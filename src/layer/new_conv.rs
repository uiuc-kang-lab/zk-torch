use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{s, Array1, ArrayD};
use rayon::prelude::*; // Import Rayon traits
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub fn get_kernel_copy_array(
  input_shape: &Vec<usize>,
  kernel_shape: &Vec<usize>,
  strides: &Vec<usize>,
  paddings: &Vec<usize>,
) -> (ArrayD<Option<IxDyn>>, usize) {
  let input_height = input_shape[0];
  let input_width = input_shape[1];
  let kernel_height = kernel_shape[0];
  let kernel_width = kernel_shape[1];
  let stride_height = strides[0];
  let stride_width = strides[1];
  let padding_top = paddings[0];
  let padding_left = paddings[1];
  let padding_bottom = paddings[2];
  let padding_right = paddings[3];

  // Calculate the dimensions of the output matrix
  let output_height = ((input_height + padding_top + padding_bottom - kernel_height) / stride_height) + 1;
  let output_width = ((input_width + padding_left + padding_right - kernel_width) / stride_width) + 1;

  // Total number of input positions
  let input_size = input_height * input_width;
  // Total number of output positions
  let output_size = output_height * output_width;

  // Initialize copy_matrix as a Vec of length output_size * input_size
  let mut copy_matrix = vec![None; output_size * input_size];

  // Parallelize over output positions using Rayon
  copy_matrix.par_chunks_mut(input_size).enumerate().for_each(|(output_idx, chunk)| {
    let i_o = output_idx / output_width;
    let j_o = output_idx % output_width;

    // Compute the top-left corner of the kernel in the input
    let start_i = (i_o * stride_height) as isize - padding_top as isize;
    let start_j = (j_o * stride_width) as isize - padding_left as isize;

    // For each position in the kernel
    for k_i in 0..kernel_height {
      for k_j in 0..kernel_width {
        let i_i = start_i + k_i as isize;
        let j_i = start_j + k_j as isize;

        // Check if the position is within the input bounds
        if i_i >= 0 && i_i < input_height as isize && j_i >= 0 && j_i < input_width as isize {
          let input_idx = (i_i as usize) * input_width + (j_i as usize);

          // The index in the kernel is (k_i, k_j)
          chunk[input_idx] = Some(IxDyn(&[k_i, k_j]));
        }
      }
    }
  });

  // Convert copy_matrix to ArrayD
  let copy_matrix_array = ArrayD::from_shape_vec(IxDyn(&[output_size, input_size]), copy_matrix).unwrap();

  // Pad to power of two
  let copy_matrix_array = pad_to_pow_of_two(&copy_matrix_array, &None);

  // Reverse axes
  let copy_matrix_array = copy_matrix_array.reversed_axes();

  (copy_matrix_array, output_size)
}

pub struct Conv2dLayer;

impl Layer for Conv2dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let weight_shape = input_shapes[1];
    let ch_dims = weight_shape[2..].to_vec();

    let orig_input_shape: Vec<usize> = match attributes.iter().find(|x| x.name == "orig_input_shape") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => panic!("orig_input_shape not found"),
    };
    let dims = orig_input_shape[2..].to_vec();

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; dims.len()],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 2 * dims.len()],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
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

    // Split
    let split_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: 1,
      split: vec![1; util::next_pow(input_shapes[0][1] as u32) as usize],
    }));
    let split_output = graph.addNode(split_bb, vec![(-1, 0)]);

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
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

    // Matmul
    let kernel = constants[1].unwrap().0;
    let (copy_array, output_dim) = get_kernel_copy_array(&dims, &ch_dims, &strides, &pads);
    let mut c_outs = Vec::new();

    for c_out in 0..weight_shape[0] {
      let mut cqlin_outputs = Vec::new();
      for c_in in 0..weight_shape[1] {
        let k = kernel.slice(s![c_out, c_in, .., ..]).into_dyn().to_owned();
        let cqlin_setup = ArrayD::from_shape_fn(copy_array.shape(), |i| if let Some(idx) = &copy_array[&i] { k[idx] } else { Fr::zero() });
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQLinBasicBlock { setup: cqlin_setup }),
          N: 1,
        }));
        let cqlin_output = graph.addNode(cqlin, vec![(split_output, c_in)]);
        cqlin_outputs.push(cqlin_output);
      }
      while cqlin_outputs.len() > 1 {
        let add_output = graph.addNode(add, vec![(cqlin_outputs.pop().unwrap(), 0), (cqlin_outputs.pop().unwrap(), 0)]);
        cqlin_outputs.push(add_output);
      }
      let add_output = cqlin_outputs.pop().unwrap();

      // Change SF
      let change_SF_output = graph.addNode(change_SF, vec![(add_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(add_output, 0), (change_SF_output, 0)]);

      let mut c_output = change_SF_output;
      // Add bias
      if constants.len() > 2 {
        let b = constants[2].unwrap().0.slice(s![c_out]).into_dyn().to_owned();
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
        let bias_output = graph.addNode(bias, vec![]);
        c_output = graph.addNode(add, vec![(change_SF_output, 0), (bias_output, 0)]);
      }

      c_outs.push(c_output);
    }

    for _ in 0..util::next_pow(weight_shape[0] as u32) as usize - weight_shape[0] {
      let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::zero(),
        shape: vec![1, 1, util::next_pow(output_dim as u32) as usize],
      }));
      let c_out_pad = graph.addNode(constantOfShape, vec![]);
      c_outs.push(c_out_pad);
    }

    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 1 }));
    let output = graph.addNode(concat, c_outs.iter().map(|&c_out| (c_out, 0)).collect());
    let output_shape = vec![1, weight_shape[0], output_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
