use crate::basic_block::*;
use crate::graph::*;
use crate::layer::new_conv::get_kernel_map;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use ark_bn254::Fr;
use ark_std::{One, Zero};
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{s, Array1, Array2, Array4, ArrayD};
use std::sync::Arc;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

#[derive(Debug)]
pub struct MaxPool2dBasicBlock {
  pub input_shape: Vec<usize>,
  pub kernel_shape: Vec<usize>,
  pub strides: Vec<usize>,
  pub pads: Vec<usize>,
}
impl BasicBlock for MaxPool2dBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let input = inputs[0].to_owned();
    let real_input_shape = vec![self.input_shape[0] * self.input_shape[1]];
    let real_input = util::slice_nd_array(input, &real_input_shape);
    let real_input_2d = real_input.into_shape(vec![self.input_shape[0], self.input_shape[1]]).unwrap();
    let output_h = (self.input_shape[0] - self.kernel_shape[0] + self.pads[0] + self.pads[2]) / self.strides[0] + 1;
    let output_w = (self.input_shape[1] - self.kernel_shape[1] + self.pads[1] + self.pads[3]) / self.strides[1] + 1;
    let mut output = Array2::<Fr>::zeros((output_h, output_w));
    // Apply padding by creating a padded view of the input
    let padded_height = self.input_shape[0] + self.pads[0] + self.pads[2];
    let padded_width = self.input_shape[1] + self.pads[1] + self.pads[3];
    let mut padded_input = Array2::<Fr>::zeros((padded_height, padded_width));
    padded_input
      .slice_mut(s![
        self.pads[0]..self.pads[0] + self.input_shape[0],
        self.pads[1]..self.pads[1] + self.input_shape[1]
      ])
      .assign(&real_input_2d);

    let cq_range_lower = *onnx::CQ_RANGE_LOWER;
    let cq_max = Fr::from(-cq_range_lower);
    let cq_min = Fr::from(cq_range_lower);
    for i in 0..output_h {
      for j in 0..output_w {
        // Define the current window based on kernel and strides
        let start_y = i * self.strides[0];
        let start_x = j * self.strides[1];

        // Extract the kernel window
        let window = padded_input.slice(s![start_y..start_y + self.kernel_shape[0], start_x..start_x + self.kernel_shape[1]]);

        // Find the maximum value in the window
        let max_value = window.iter().cloned().fold(cq_min, |max, y| {
          if util::fr_to_int(y) < util::fr_to_int(cq_max) && util::fr_to_int(y) > util::fr_to_int(max) {
            y
          } else {
            max
          }
        });
        output[(i, j)] = max_value;
      }
    }
    let real_output = output.into_dyn().into_shape(vec![output_h * output_w]).unwrap();
    let real_output = util::pad_to_pow_of_two(&real_output, &Fr::zero());

    Ok(vec![real_output])
  }
}

#[derive(Debug)]
pub struct CreateSelectorBasicBlock;
impl BasicBlock for CreateSelectorBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let max_pool_output = inputs[0];
    let mut outputs = vec![Array1::<Fr>::zeros(max_pool_output.len()); inputs.len() - 1];
    max_pool_output.iter().enumerate().for_each(|(i, &num)| {
      for j in 1..inputs.len() {
        let compared = inputs[j].clone().into_raw_vec()[i];
        if num == compared {
          outputs[j - 1][i] = Fr::one();
        }
        break;
      }
    });
    let outputs = outputs.into_iter().map(|x| x.into_dyn()).collect();
    Ok(outputs)
  }
}

pub struct MaxPool2dLayer;

impl Layer for MaxPool2dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let ch_dims: Vec<usize> = match attributes.iter().filter(|x| x.name == "kernel_shape").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => panic!("kernel_shape not found"),
    };

    let in_channel: usize = match attributes.iter().filter(|x| x.name == "in_channel").next() {
      Some(v) => v.i as usize,
      None => panic!("in_channel not found"),
    };

    // only support square image for now
    let input_h = ((input_shapes[0][1] / in_channel) as f64).sqrt() as usize;
    let dims = vec![input_h, input_h];

    let strides = match attributes.iter().filter(|x| x.name == "strides").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 2],
    };
    let pads = match attributes.iter().filter(|x| x.name == "pads").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 4],
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

    let (mut a, mut b) = (input_shapes[0][1] / in_channel, in_channel);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation_1 = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    let transpose_1 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation_1 }),
      N: 2,
    }));
    let transpose_output_1 = graph.addNode(transpose_1, vec![(-1, 0)]);

    // MaxPool2d
    let maxpool = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MaxPool2dBasicBlock {
        input_shape: dims.clone(),
        kernel_shape: ch_dims.clone(),
        strides: strides.clone(),
        pads: pads.clone(),
      }),
      N: 1,
    }));
    let maxpool_output = graph.addNode(maxpool, vec![(transpose_output_1, 0)]);

    // Sub
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let multi_add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MultipleAddBasicBlock {}),
      N: 1,
    }));
    let create_selector = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CreateSelectorBasicBlock {}),
      N: 1,
    }));
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));

    // CQ to check if x >= 0
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    graph.shared_kernels = Some(Vec::new());
    // Matmul
    let (copy_array, output_dim) = get_kernel_map(&vec![1, dims[0], dims[1]], &vec![1, 1, ch_dims[0], ch_dims[1]], &strides, &pads);
    let mut out_and_selectors = vec![(maxpool_output, 0)];
    for idx in 0..ch_dims[0] * ch_dims[1] {
      let mut k = Array4::<i128>::zeros((1, 1, ch_dims[0], ch_dims[1]));
      k[(0, 0, idx / ch_dims[1], idx % ch_dims[1])] = 1;
      graph.shared_kernels.as_mut().unwrap().push(Arc::new(k.into_dyn()));
      let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(SparseCQLinBasicBlock {
          kernel: Arc::clone(&graph.shared_kernels.as_ref().unwrap()[idx]),
          indices: copy_array.clone(),
          input_len: util::next_pow((dims[0] * dims[1]) as u32) as usize,
          output_len: util::next_pow((output_dim) as u32) as usize,
        }),
        N: 2,
      }));
      let cqlin_output = graph.addNode(cqlin, vec![(transpose_output_1, 0)]);
      let sub_output = graph.addNode(sub, vec![(maxpool_output, 0), (cqlin_output, 0)]);
      // Check if the remainder sub_output is non-negative
      let _ = graph.addNode(range_check, vec![(sub_output, 0)]);
      out_and_selectors.push((cqlin_output, 0));
    }
    let selector_output = graph.addNode(create_selector, out_and_selectors.clone());

    let mut to_sum = Vec::new();
    for idx in 0..ch_dims[0] * ch_dims[1] {
      let _ = graph.addNode(bool_check, vec![(selector_output, idx)]);
      let mul_output = graph.addNode(mul, vec![(selector_output, idx), out_and_selectors[idx + 1]]);
      to_sum.push((mul_output, 0));
    }
    let add_output = graph.addNode(multi_add, to_sum);
    let _ = graph.addNode(eq, vec![(add_output, 0), (maxpool_output, 0)]);

    let c = in_channel * output_dim;
    let c = util::next_pow(c as u32) as usize;
    let permutation = (vec![0], (0..c).collect());
    let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation }),
      N: 2,
    }));
    let maxpool_output = graph.addNode(transpose, vec![(maxpool_output, 0)]);

    let output_shape = vec![1, in_channel * output_dim];
    graph.outputs.push((maxpool_output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
