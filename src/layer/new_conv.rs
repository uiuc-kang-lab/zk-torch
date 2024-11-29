use crate::basic_block::*;
use crate::graph::*;
use crate::layer::new_conv::cqlin::KernelIdx;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use crate::CONFIG;
use ark_bn254::Fr;
use ark_std::{One, Zero};
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{s, Array1, Array4, ArrayD, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;
use tract_onnx::prelude::Outlet; // Import Rayon traits

pub fn get_kernel_map(
  input_shape: &Vec<usize>,
  kernel_shape: &Vec<usize>,
  strides: &Vec<usize>,
  paddings: &Vec<usize>,
) -> (HashMap<(usize, usize), KernelIdx>, usize) {
  let input_channel = input_shape[0];
  let input_height = input_shape[1];
  let input_width = input_shape[2];
  let kernel_out_channel = kernel_shape[0];
  let kernel_in_channel = kernel_shape[1];
  let kernel_height = kernel_shape[2];
  let kernel_width = kernel_shape[3];
  assert!(kernel_in_channel == input_channel);

  // Calculate the dimensions of the output matrix
  let output_height = ((input_height - kernel_height + paddings[0] + paddings[2]) / strides[0]) + 1;
  let output_width = ((input_width - kernel_width + paddings[1] + paddings[3]) / strides[1]) + 1;

  // Create a padded input matrix in parallel
  let padded_matrix: Vec<Vec<Vec<i32>>> = (0..input_channel)
    .into_par_iter()
    .map(|c| {
      let mut channel_matrix = vec![vec![-1; input_width + paddings[1] + paddings[3]]; input_height + paddings[0] + paddings[2]];
      for i in 0..input_height {
        for j in 0..input_width {
          let value = (c * input_height * input_width + i * input_width + j) as i32;
          channel_matrix[i + paddings[0]][j + paddings[1]] = value;
        }
      }
      channel_matrix
    })
    .collect();

  // Calculate total number of output elements
  let total_outputs = kernel_out_channel * output_height * output_width;

  // Generate all indices in parallel
  let mut kernel_hashmap = HashMap::new();

  let _ = (0..total_outputs).into_iter().map(|idx| {
    let co = idx / (output_height * output_width);
    let remaining = idx % (output_height * output_width);
    let i = remaining / output_width;
    let j = remaining % output_width;

    for c in 0..kernel_in_channel {
      for ki in 0..kernel_height {
        for kj in 0..kernel_width {
          let input_value = padded_matrix[c][i * strides[0] + ki][j * strides[1] + kj];
          if input_value != -1 {
            let key = KernelIdx::TwoD(co, c, ki, kj);
            kernel_hashmap.insert((idx, input_value as usize), key);
          }
        }
      }
    }
  });

  (kernel_hashmap, total_outputs)
}

pub fn get_kernel_map_3d(
  input_shape: &Vec<usize>,
  kernel_shape: &Vec<usize>,
  strides: &Vec<usize>,
  paddings: &Vec<usize>,
) -> (HashMap<(usize, usize), KernelIdx>, usize) {
  let input_channel = input_shape[0];
  let input_d1 = input_shape[1];
  let input_d2 = input_shape[2];
  let input_d3 = input_shape[3];
  let kernel_out_channel = kernel_shape[0];
  let kernel_in_channel = kernel_shape[1];
  let kernel_d1 = kernel_shape[2];
  let kernel_d2 = kernel_shape[3];
  let kernel_d3 = kernel_shape[4];
  assert!(kernel_in_channel == input_channel);

  // Calculate the dimensions of the output matrix
  let output_d1 = ((input_d1 - kernel_d1 + paddings[0] + paddings[3]) / strides[0]) + 1;
  let output_d2 = ((input_d2 - kernel_d2 + paddings[1] + paddings[4]) / strides[1]) + 1;
  let output_d3 = ((input_d3 - kernel_d3 + paddings[2] + paddings[5]) / strides[2]) + 1;

  // Create a padded input matrix in parallel
  let padded_matrix: Vec<Vec<Vec<Vec<i32>>>> = (0..input_channel)
    .into_par_iter()
    .map(|c| {
      let mut channel_matrix =
        vec![vec![vec![-1; input_d3 + paddings[2] + paddings[5]]; input_d2 + paddings[1] + paddings[4]]; input_d1 + paddings[0] + paddings[3]];
      for i in 0..input_d1 {
        for j in 0..input_d2 {
          for k in 0..input_d3 {
            let value = (c * input_d1 * input_d2 * input_d3 + i * input_d2 * input_d3 + j * input_d3 + k) as i32;
            channel_matrix[i + paddings[0]][j + paddings[1]][k + paddings[2]] = value;
          }
        }
      }
      channel_matrix
    })
    .collect();

  // Calculate total number of output elements
  let total_outputs = kernel_out_channel * output_d1 * output_d2 * output_d3;

  // Generate all indices in parallel
  let mut kernel_hashmap = HashMap::new();

  let _ = (0..total_outputs).into_iter().map(|idx| {
    let co = idx / (output_d1 * output_d2 * output_d3);
    let remaining = idx % (output_d1 * output_d2 * output_d3);
    let i = remaining / (output_d2 * output_d3);
    let j = remaining / output_d3;
    let k = remaining % output_d3;

    for c in 0..kernel_in_channel {
      for ki in 0..kernel_d1 {
        for kj in 0..kernel_d2 {
          for kk in 0..kernel_d3 {
            let input_value = padded_matrix[c][i * strides[0] + ki][j * strides[1] + kj][k * strides[2] + kk];
            if input_value != -1 {
              let key = KernelIdx::ThreeD(co, c, ki, kj, kk);
              kernel_hashmap.insert((idx, input_value as usize), key);
            }
          }
        }
      }
    }
  });

  (kernel_hashmap, total_outputs)
}

pub fn slice_kernel_map(
  copy_array: &HashMap<(usize, usize), KernelIdx>,
  inp_start: usize,
  inp_end: usize,
  out_start: usize,
  out_end: usize,
) -> HashMap<(usize, usize), KernelIdx> {
  let mut output = copy_array.clone();
  for (key, value) in copy_array.iter() {
    let (out_idx, in_idx) = key;
    output.remove(key);
    if *in_idx >= inp_start && *in_idx < inp_end && *out_idx >= out_start && *out_idx < out_end {
      output.insert((*out_idx - out_start, *in_idx - inp_start), *value);
    }
  }
  output
}

// return log2(slice_input_len), log2(slice_output_len)
pub fn compute_optimal_division() -> (usize, usize) {
  let ptau_len = CONFIG.ptau.loaded_pow_len_log;
  if ptau_len % 2 == 0 {
    ((ptau_len - 4) / 2, (ptau_len + 2) / 2)
  } else {
    ((ptau_len - 3) / 2, (ptau_len + 1) / 2)
  }
}

// input: [B, C * H * W]
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
    let ch_dims = weight_shape.to_vec();

    // only support square image for now
    let image_h = ((input_shape[1] / ch_dims[1]) as f64).sqrt() as usize;
    let dims = vec![ch_dims[1], image_h, image_h];

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 2],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 4],
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
    let multi_add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MultipleAddBasicBlock {}),
      N: 1,
    }));

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

    // 1x1 kernel
    let mut shared_k_len = 0;
    graph.shared_kernels = Some(Vec::new());
    if ch_dims[2] == 1 && ch_dims[3] == 1 {
      let mut input_index = -1;
      if strides[0] == 2 && strides[1] == 2 {
        let mut b = ch_dims[1];
        let mut a = input_shapes[0][1] / b;
        a = util::next_pow(a as u32) as usize;
        b = util::next_pow(b as u32) as usize;
        let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
        let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(PermuteBasicBlock { permutation }),
          N: 2,
        }));
        input_index = graph.addNode(transpose, vec![(input_index, 0)]);
        let local_pad = vec![0, 0, 0, 0];
        let (local_copy_map, local_output_len) = get_kernel_map(&vec![1, dims[1], dims[2]], &vec![1, 1, 2, 2], &strides, &local_pad);
        let local_input_len = util::next_pow((dims[1] * dims[2]) as u32) as usize;
        let local_output_len = util::next_pow(local_output_len as u32) as usize;
        let mut k = Array4::<i128>::zeros((1, 1, 2, 2));
        k[(0, 0, 0, 0)] = 1;
        graph.shared_kernels.as_mut().unwrap().push(Arc::new(k.into_dyn()));
        shared_k_len += 1;
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: Arc::clone(&graph.shared_kernels.as_ref().unwrap()[shared_k_len - 1]),
            indices: local_copy_map,
            input_len: local_input_len,
            output_len: local_output_len,
          }),
          N: 2,
        }));
        input_index = graph.addNode(cqlin, vec![(input_index, 0)]);
      }
      let (mut a, mut b) = (ch_dims[1], (dims[1] * dims[2]) / (strides[0] * strides[1]));
      let mut c = weight_shape[0] * (dims[1] * dims[2]) / (strides[0] * strides[1]);
      a = util::next_pow(a as u32) as usize;
      b = util::next_pow(b as u32) as usize;
      c = util::next_pow(c as u32) as usize;
      let permutation_1 = ((0..b).map(|x| x * a).collect(), (0..a).collect());
      let permutation_2 = (vec![0], (0..c).collect());
      let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_1 }),
        N: 2,
      }));
      let k = constants[1].unwrap().0.clone().into_shape((weight_shape[0], weight_shape[1])).unwrap();
      let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQLinBasicBlock {
          setup: k.t().to_owned().into_dyn(),
        }),
        N: 2,
      }));
      let transpose_output = graph.addNode(transpose, vec![(input_index, 0)]);
      let cqlin_output = graph.addNode(cqlin, vec![(transpose_output, 0)]);
      let mut output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(cqlin_output, 0), (output, 0)]);
      if input_shapes.len() > 2 {
        let b = constants[2].unwrap().0;
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b.clone() }));
        let bias_output = graph.addNode(bias, vec![]);
        output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
      }
      let transpose_2 = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_2 }),
        N: 2,
      }));
      let output = graph.addNode(transpose_2, vec![(output, 0)]);

      graph.outputs.push((output, 0));
      let output_shape = vec![1, weight_shape[0] * (dims[1] * dims[2]) / (strides[0] * strides[1])];
      return (graph, vec![output_shape], vec![input_types[0]]);
    }

    // Matmul
    let (copy_map, output_dim) = get_kernel_map(&dims, &ch_dims, &strides, &pads);
    let input_len = util::next_pow((dims[0] * dims[1] * dims[2]) as u32) as usize;
    let output_len = util::next_pow(output_dim as u32) as usize;

    let (m, n) = compute_optimal_division();
    let (slice_input_len, slice_output_len) = (1 << m, 1 << n);
    let (slice_input_num, slice_output_num) = (input_len / slice_input_len, output_len / slice_output_len);

    let mut input_index = -1;
    if input_len > slice_input_len {
      let a = slice_input_len;
      let b = slice_input_num;
      let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
      let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation }),
        N: 2,
      }));
      input_index = graph.addNode(transpose, vec![(input_index, 0)]);
    }

    // Split
    let split_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: 0,
      split: vec![1; slice_input_num],
    }));
    let split_output = graph.addNode(split_bb, vec![(input_index, 0)]);

    let mut add_outputs = Vec::new();
    let kernel = constants[1].unwrap().0.clone().map(|x| util::fr_to_int(*x)).into_dyn();
    graph.shared_kernels.as_mut().unwrap().push(Arc::new(kernel.clone()));
    shared_k_len += 1;
    for n in 0..slice_output_num {
      let mut cqlin_outputs = Vec::new();
      for m in 0..slice_input_num {
        let copy = slice_kernel_map(
          &copy_map,
          slice_input_len * m,
          slice_input_len * (m + 1),
          slice_output_len * n,
          slice_output_len * (n + 1),
        );

        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: Arc::clone(&graph.shared_kernels.as_ref().unwrap()[shared_k_len - 1]),
            indices: copy,
            input_len: slice_input_len,
            output_len: slice_output_len,
          }),
          N: 1,
        }));
        let cqlin_output = graph.addNode(cqlin, vec![(split_output, m)]);
        cqlin_outputs.push(cqlin_output);
      }
      let add_output = graph.addNode(multi_add, cqlin_outputs.iter().map(|&c_out| (c_out, 0)).collect());
      add_outputs.push(add_output);
    }

    // Concat
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 0 }));
    let concat_output = graph.addNode(concat, add_outputs.iter().map(|&a_out| (a_out, 0)).collect());

    // Permute
    let permutation = (vec![0], (0..util::next_pow(output_len as u32) as usize).collect());
    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation }),
      N: 2,
    }));
    let transpose_output = graph.addNode(transpose, vec![(concat_output, 0)]);

    // Change SF
    let mut output = graph.addNode(change_SF, vec![(transpose_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(transpose_output, 0), (output, 0)]);

    // Add bias
    if input_shapes.len() > 2 {
      let b = util::slice_nd_array(constants[2].unwrap().0.to_owned(), &vec![ch_dims[0]]);
      let b_len = b.len();
      let reshaped = b.into_shape((b_len, 1)).unwrap(); // Reshape to [C, 1]
      let repeated = reshaped.broadcast((b_len, output_dim / b_len)).unwrap(); // Broadcast to [C, K]
      let flattened = repeated.to_owned().into_shape((output_dim,)).unwrap(); // Flatten to [K * C]
      let b = util::pad_to_pow_of_two(&flattened.into_dyn(), &Fr::zero());
      let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
      let bias_output = graph.addNode(bias, vec![]);
      output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
    }

    let output_shape = vec![1, output_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

// input shape: [B, C, D1*D2*D3] (original: [B, C, D1, D2, D3])
pub struct Conv3dLayer;

impl Layer for Conv3dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0];
    let weight_shape = input_shapes[1];
    let ch_dims = weight_shape.to_vec();

    // only support cube image for now
    let image_dim = (input_shape[2] as f64).cbrt() as usize;
    let dims = vec![ch_dims[1], image_dim, image_dim, image_dim];

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 3],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 6],
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
      None => vec![1; 3],
    };

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let multi_add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MultipleAddBasicBlock {}),
      N: 1,
    }));

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

    // TODO: 1x1x1 kernel
    let mut shared_k_len = 0;
    graph.shared_kernels = Some(Vec::new());
    if ch_dims[2] == 1 && ch_dims[3] == 1 && ch_dims[4] == 1 {
      let mut input_index = -1;
      if strides[0] == 2 && strides[1] == 2 {
        let mut b = ch_dims[1];
        let mut a = input_shapes[0][1] / b;
        a = util::next_pow(a as u32) as usize;
        b = util::next_pow(b as u32) as usize;
        let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
        let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(PermuteBasicBlock { permutation }),
          N: 2,
        }));
        input_index = graph.addNode(transpose, vec![(input_index, 0)]);
        let local_pad = vec![0, 0, 0, 0];
        let (local_copy_map, local_output_len) = get_kernel_map(&vec![1, dims[1], dims[2]], &vec![1, 1, 2, 2], &strides, &local_pad);
        let local_input_len = util::next_pow((dims[1] * dims[2]) as u32) as usize;
        let local_output_len = util::next_pow(local_output_len as u32) as usize;
        let mut k = Array4::<i128>::zeros((1, 1, 2, 2));
        k[(0, 0, 0, 0)] = 1;
        graph.shared_kernels.as_mut().unwrap().push(Arc::new(k.into_dyn()));
        shared_k_len += 1;
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: Arc::clone(&graph.shared_kernels.as_ref().unwrap()[shared_k_len - 1]),
            indices: local_copy_map,
            input_len: local_input_len,
            output_len: local_output_len,
          }),
          N: 2,
        }));
        input_index = graph.addNode(cqlin, vec![(input_index, 0)]);
      }
      let (mut a, mut b) = (ch_dims[1], (dims[1] * dims[2]) / (strides[0] * strides[1]));
      let mut c = weight_shape[0] * (dims[1] * dims[2]) / (strides[0] * strides[1]);
      a = util::next_pow(a as u32) as usize;
      b = util::next_pow(b as u32) as usize;
      c = util::next_pow(c as u32) as usize;
      let permutation_1 = ((0..b).map(|x| x * a).collect(), (0..a).collect());
      let permutation_2 = (vec![0], (0..c).collect());
      let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_1 }),
        N: 2,
      }));
      let k = constants[1].unwrap().0.clone().into_shape((weight_shape[0], weight_shape[1])).unwrap();
      let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(CQLinBasicBlock {
          setup: k.t().to_owned().into_dyn(),
        }),
        N: 2,
      }));
      let transpose_output = graph.addNode(transpose, vec![(input_index, 0)]);
      let cqlin_output = graph.addNode(cqlin, vec![(transpose_output, 0)]);
      let mut output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(cqlin_output, 0), (output, 0)]);
      if input_shapes.len() > 2 {
        let b = constants[2].unwrap().0;
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b.clone() }));
        let bias_output = graph.addNode(bias, vec![]);
        output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
      }
      let transpose_2 = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation_2 }),
        N: 2,
      }));
      let output = graph.addNode(transpose_2, vec![(output, 0)]);

      graph.outputs.push((output, 0));
      let output_shape = vec![1, weight_shape[0] * (dims[1] * dims[2]) / (strides[0] * strides[1])];
      return (graph, vec![output_shape], vec![input_types[0]]);
    }

    // Matmul
    let (copy_map, output_dim) = get_kernel_map_3d(&dims, &ch_dims, &strides, &pads);
    let input_len = util::next_pow((dims[0] * dims[1] * dims[2] * dims[3]) as u32) as usize;
    let output_len = util::next_pow(output_dim as u32) as usize;

    // Permute first
    let mut input_index = -1;
    let permutation = (vec![0], (0..input_len).collect());
    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation }),
      N: 2,
    }));
    input_index = graph.addNode(transpose, vec![(input_index, 0)]); // shape: [1, 1, input_len]

    let (m, n) = compute_optimal_division();
    let (slice_input_len, slice_output_len) = (1 << m, 1 << n);
    let (slice_input_num, slice_output_num) = (input_len / slice_input_len, output_len / slice_output_len);

    if input_len > slice_input_len {
      let a = slice_input_len;
      let b = slice_input_num;
      let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
      let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation }),
        N: 2,
      }));
      input_index = graph.addNode(transpose, vec![(input_index, 0)]); // shape: [1, slice_input_num, slice_input_len]
    }

    // Split
    let split_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: 1,
      split: vec![1; slice_input_num],
    }));
    let split_output = graph.addNode(split_bb, vec![(input_index, 0)]); // shape: [1, 1, slice_input_len]

    let mut add_outputs = Vec::new();
    let kernel = constants[1].unwrap().0.clone().map(|x| util::fr_to_int(*x)).into_dyn();
    graph.shared_kernels.as_mut().unwrap().push(Arc::new(kernel.clone()));
    shared_k_len += 1;
    for n in 0..slice_output_num {
      let mut cqlin_outputs = Vec::new();
      for m in 0..slice_input_num {
        let copy = slice_kernel_map(
          &copy_map,
          slice_input_len * m,
          slice_input_len * (m + 1),
          slice_output_len * n,
          slice_output_len * (n + 1),
        );

        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: Arc::clone(&graph.shared_kernels.as_ref().unwrap()[shared_k_len - 1]),
            indices: copy,
            input_len: slice_input_len,
            output_len: slice_output_len,
          }),
          N: 1,
        }));
        let cqlin_output = graph.addNode(cqlin, vec![(split_output, m)]); // shape: [1, 1, slice_output_len]
        cqlin_outputs.push(cqlin_output);
      }
      let add_output = graph.addNode(multi_add, cqlin_outputs.iter().map(|&c_out| (c_out, 0)).collect()); // shape: [1, 1, slice_output_len]
      add_outputs.push(add_output);
    }

    // Concat
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 1 }));
    let concat_output = graph.addNode(concat, add_outputs.iter().map(|&a_out| (a_out, 0)).collect()); // shape: [1, slice_output_num, slice_output_len]

    // Permute back
    let b = util::next_pow(ch_dims[0] as u32) as usize;
    let a = util::next_pow(output_len as u32) as usize / b;

    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation }),
      N: 2,
    }));
    let transpose_output = graph.addNode(transpose, vec![(concat_output, 0)]); // shape: [1, ch_dims[0], output_len / ch_dims[0]]

    // Change SF
    let mut output = graph.addNode(change_SF, vec![(transpose_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(transpose_output, 0), (output, 0)]);

    // Add bias
    if input_shapes.len() > 2 {
      let b = util::slice_nd_array(constants[2].unwrap().0.to_owned(), &vec![ch_dims[0]]);
      let b_len = b.len();
      let reshaped = b.into_shape((b_len, 1)).unwrap(); // Reshape to [C, 1]
      let repeated = reshaped.broadcast((b_len, output_dim / b_len)).unwrap(); // Broadcast to [C, K]
      let flattened = repeated.to_owned().into_shape((1, b_len, output_dim / b_len)).unwrap(); // into [1, C, K]
      let b = util::pad_to_pow_of_two(&flattened.into_dyn(), &Fr::zero());
      let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
      let bias_output = graph.addNode(bias, vec![]);
      output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
    }

    let output_shape = vec![1, ch_dims[0], output_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

pub struct Conv3dTransposeLayer;
impl Layer for Conv3dTransposeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let weight_shape = input_shapes[1];
    let ch_dims = weight_shape.to_vec();

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 3],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 6],
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
      None => vec![1; 3],
    };

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

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

    // only support 2x2x2 kernel, 2x2x2 stride, and 1x1x1x1x1x1 padding for now
    assert!(ch_dims[2] == 2 && ch_dims[3] == 2 && ch_dims[4] == 2);
    assert!(strides[0] == 2 && strides[1] == 2 && strides[2] == 2);
    assert!(pads[0] == 1 && pads[1] == 1 && pads[2] == 1 && pads[3] == 1 && pads[4] == 1 && pads[5] == 1);

    let mut a = input_shapes[0][0];
    let mut b = input_shapes[0][1];
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation }),
      N: 2,
    }));
    let k = util::pad_to_pow_of_two(constants[1].unwrap().0, &Fr::zero());
    let k_shape_0 = a;
    let k_shape_1 = util::next_pow((ch_dims[1] * ch_dims[2] * ch_dims[3] * ch_dims[4]) as u32) as usize;

    let k = k.into_shape((k_shape_0, k_shape_1)).unwrap();
    let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQLinBasicBlock { setup: k.into_dyn() }),
      N: 2,
    }));

    let mut c = ch_dims[1];
    let mut d = input_shapes[0][1] * ch_dims[2] * ch_dims[3] * ch_dims[4];
    c = util::next_pow(c as u32) as usize;
    d = util::next_pow(d as u32) as usize;
    let permutation_1 = ((0..c).map(|x| x * d).collect(), (0..d).collect());
    let transpose_1 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation_1 }),
      N: 2,
    }));

    let transpose_output = graph.addNode(transpose, vec![(-1, 0)]);
    let cqlin_output = graph.addNode(cqlin, vec![(transpose_output, 0)]);
    let change_sf_output = graph.addNode(change_SF, vec![(cqlin_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(cqlin_output, 0), (change_sf_output, 0)]);
    let mut output = graph.addNode(transpose_1, vec![(change_sf_output, 0)]);

    // Add bias
    if input_shapes.len() > 2 {
      let b = util::slice_nd_array(constants[2].unwrap().0.to_owned(), &vec![ch_dims[1]]);
      let b_len = b.len();
      let reshaped = b.into_shape((b_len, 1)).unwrap(); // Reshape to [C, 1]
      let b = util::pad_to_pow_of_two(&reshaped.into_dyn(), &Fr::zero());
      let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
      let bias_output = graph.addNode(bias, vec![]);
      output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
    }

    let output_shape = vec![ch_dims[1], input_shapes[0][1] * ch_dims[2] * ch_dims[3] * ch_dims[4]];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
