use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use crate::CONFIG;
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
) -> (Vec<Vec<Option<usize>>>, usize) {
  let input_height = input_shape[0];
  let input_width = input_shape[1];
  let kernel_height = kernel_shape[0];
  let kernel_width = kernel_shape[1];
  // Calculate the dimensions of the output matrix
  let output_height = ((input_height - kernel_height + paddings[0] + paddings[2]) / strides[0]) + 1;
  let output_width = ((input_width - kernel_width + paddings[1] + paddings[3]) / strides[1]) + 1;
  // Create a padded input matrix
  let mut padded_matrix = vec![vec![-1; input_width + paddings[1] + paddings[3]]; input_height + paddings[0] + paddings[2]];
  let mut value = 0;
  for i in 0..input_height {
    for j in 0..input_width {
      padded_matrix[i + paddings[0]][j + paddings[1]] = value;
      value += 1;
    }
  }
  // Initialize the output matrix
  let mut output = Vec::new();
  // Perform the 2D convolution
  for i in 0..output_height {
    for j in 0..output_width {
      let mut kernel_vec = Vec::new();
      for ki in 0..kernel_height {
        for kj in 0..kernel_width {
          let input_value = padded_matrix[i * strides[0] + ki][j * strides[1] + kj];
          if input_value != -1 {
            kernel_vec.push(Some(input_value as usize));
          } else {
            kernel_vec.push(None);
          }
        }
      }
      output.push(kernel_vec);
    }
  }
  // pad to pow of 2
  let pad_output_len = util::next_pow(output.len() as u32) as usize;
  let output_len = output.len();
  for _ in 0..pad_output_len - output_len {
    output.push(vec![None; kernel_height * kernel_width]);
  }

  (output, output_len)
}

pub fn split_copy_array(copy_array: Vec<Vec<Option<usize>>>, pieces: usize, cut_dim: usize) -> Vec<Vec<Vec<Option<usize>>>> {
  let mut output = Vec::new();
  for p in 0..pieces {
    let mut piece = Vec::new();
    for i in 0..copy_array.len() {
      let mut row = Vec::new();
      for j in 0..copy_array[i].len() {
        if let Some(v) = copy_array[i][j] {
          if v / cut_dim == p {
            row.push(Some(v % cut_dim));
          } else {
            row.push(None);
          }
        } else {
          row.push(None);
        }
      }
      piece.push(row);
    }
    output.push(piece);
  }
  output
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
    let input_shape = input_shapes[0];
    let weight_shape = input_shapes[1];
    let ch_dims = weight_shape[2..].to_vec();

    // only support square image for now
    let dims = vec![(input_shape[2] as f64).sqrt() as usize; 2];

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
    let sf_float = onnx::SF_FLOAT.read().unwrap().to_owned();
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
    let input_len = util::next_pow((dims[0] * dims[1]) as u32) as usize;
    let output_len = copy_array.len();
    assert!((input_len * output_len) < 1 << CONFIG.ptau.loaded_pow_len_log);

    let mut c_outs = Vec::new();

    for c_out in 0..weight_shape[0] {
      let mut cqlin_outputs = Vec::new();
      for c_in in 0..weight_shape[1] {
        let k = kernel.slice(s![c_out, c_in, .., ..]).to_owned().into_raw_vec();
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: k,
            indices: copy_array.clone(),
            input_len: input_len,
          }),
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

      let mut c_output = add_output;
      // Add bias
      if constants.len() > 2 {
        let b = constants[2].unwrap().0.slice(s![c_out]).into_dyn().to_owned().map(
          |x| {
            let y = (util::fr_to_int(*x) as f32 * sf_float).round();
            Fr::from(y as i32)
          },
        );
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
        let bias_output = graph.addNode(bias, vec![]);
        c_output = graph.addNode(add, vec![(add_output, 0), (bias_output, 0)]);
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
    let concat_output = graph.addNode(concat, c_outs.iter().map(|&c_out| (c_out, 0)).collect());

    // Change SF
    let output = graph.addNode(change_SF, vec![(concat_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(concat_output, 0), (output, 0)]);

    let output_shape = vec![1, weight_shape[0], output_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

pub struct LargeConv2dLayer;

impl Layer for LargeConv2dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0];
    let weight_shape = input_shapes[1];
    let ch_dims = weight_shape[2..].to_vec();

    // only support square image for now
    let dims = vec![((input_shape[2] * input_shape[3]) as f64).sqrt() as usize; 2];

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
    let split_bb_1 = graph.addBB(Box::new(SplitBasicBlock {
      axis: 2,
      split: vec![1; util::next_pow(input_shapes[0][2] as u32) as usize],
    }));
    let squeeze_axis_2 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![1, 1, util::next_pow(input_shapes[0][3] as u32) as usize],
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let sf_float = onnx::SF_FLOAT.read().unwrap().to_owned();
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
    let sub_copy_arrays = split_copy_array(copy_array, input_shapes[0][2], input_shapes[0][3]);
    let input_len = util::next_pow(input_shapes[0][3] as u32) as usize;
    let output_len = sub_copy_arrays[0].len();
    assert!((input_len * output_len) < 1 << CONFIG.ptau.loaded_pow_len_log);

    let mut c_outs = Vec::new();

    for c_out in 0..weight_shape[0] {
      let mut cqlin_outputs = Vec::new();
      for c_in in 0..weight_shape[1] {
        let k = kernel.slice(s![c_out, c_in, .., ..]).to_owned().into_raw_vec();
        let split_output_1 = graph.addNode(split_bb_1, vec![(split_output, c_in)]);
        for (i_arr, copy_array) in sub_copy_arrays.iter().enumerate() {
          let squeeze = graph.addNode(squeeze_axis_2, vec![(split_output_1, i_arr)]);
          let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
            basic_block: Box::new(SparseCQLinBasicBlock {
              kernel: k.clone(),
              indices: copy_array.clone(),
              input_len: input_len,
            }),
            N: 1,
          }));
          let cqlin_output = graph.addNode(cqlin, vec![(squeeze, 0)]);
          cqlin_outputs.push(cqlin_output);
        }
      }
      while cqlin_outputs.len() > 1 {
        let add_output = graph.addNode(add, vec![(cqlin_outputs.pop().unwrap(), 0), (cqlin_outputs.pop().unwrap(), 0)]);
        cqlin_outputs.push(add_output);
      }
      let add_output = cqlin_outputs.pop().unwrap();

      let mut c_output = add_output;
      // Add bias
      if constants.len() > 2 {
        let b = constants[2].unwrap().0.slice(s![c_out]).into_dyn().to_owned().map(
          |x| {
            let y = (util::fr_to_int(*x) as f32 * sf_float).round();
            Fr::from(y as i32)
          },
        );
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
        let bias_output = graph.addNode(bias, vec![]);
        c_output = graph.addNode(add, vec![(add_output, 0), (bias_output, 0)]);
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
    let concat_output = graph.addNode(concat, c_outs.iter().map(|&c_out| (c_out, 0)).collect());

    // Change SF
    let output = graph.addNode(change_SF, vec![(concat_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(concat_output, 0), (output, 0)]);

    let output_shape = vec![1, weight_shape[0], output_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
