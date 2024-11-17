use crate::basic_block::*;
use crate::graph::*;
use crate::layer::new_conv::get_kernel_copy_array;
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
use ndarray::{s, Array1, Array2, ArrayD};
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

    // only support square image for now
    let dims = vec![(input_shapes[0][2] as f64).sqrt() as usize; 2];

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
    let maxpool_output = graph.addNode(maxpool, vec![(-1, 0)]);

    // Sub
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let create_selector = graph.addBB(Box::new(CreateSelectorBasicBlock {}));
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));

    // Split
    let split_bb = graph.addBB(Box::new(SplitBasicBlock {
      axis: 1 as usize,
      split: vec![1; util::next_pow(input_shapes[0][1] as u32) as usize],
    }));
    let split_output = graph.addNode(split_bb, vec![(-1, 0)]);
    let split_maxpool_output = graph.addNode(split_bb, vec![(maxpool_output, 0)]);

    // CQ to check if x >= 0
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    // Matmul
    let (copy_array, output_dim) = get_kernel_copy_array(&dims, &ch_dims, &strides, &pads);
    for c_in in 0..input_shapes[0][1] {
      let mut out_and_selectors = vec![(split_maxpool_output, c_in)];
      for idx in 0..ch_dims[0] * ch_dims[1] {
        let mut k = Array2::<Fr>::zeros((ch_dims[0], ch_dims[1]));
        k[(idx / ch_dims[1], idx % ch_dims[1])] = Fr::one();
        let k = k.into_raw_vec();
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(SparseCQLinBasicBlock {
            kernel: k,
            indices: copy_array.clone(),
            input_len: util::next_pow((dims[0] * dims[1]) as u32) as usize,
          }),
          N: 1,
        }));
        let cqlin_output = graph.addNode(cqlin, vec![(split_output, c_in)]);
        let sub_output = graph.addNode(sub, vec![(split_maxpool_output, c_in), (cqlin_output, 0)]);
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
      while to_sum.len() > 1 {
        let add_output = graph.addNode(add, vec![to_sum.pop().unwrap(), to_sum.pop().unwrap()]);
        to_sum.push((add_output, 0));
      }
      let add_output = to_sum.pop().unwrap();
      let _ = graph.addNode(eq, vec![add_output, (split_maxpool_output, c_in)]);
    }

    let output_shape = vec![1, input_shapes[0][1], output_dim];
    graph.outputs.push((maxpool_output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
