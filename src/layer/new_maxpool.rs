use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use ark_bn254::Fr;
use ark_bn254::G1Projective;
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
pub struct MaxPool2dCandidateBasicBlock {
  pub input_shape: Vec<usize>,  // [1, H_in, W_in, C_in]
  pub kernel_shape: Vec<usize>, // [k_h, k_w]
  pub strides: Vec<usize>,      // [s_h, s_w]
  pub pads: Vec<usize>,         // [p_h, p_w, p_h, p_w]
}
impl BasicBlock for MaxPool2dCandidateBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let input = inputs[0].to_owned();
    let C_in = self.input_shape[3];
    let output_h = (self.input_shape[1] - self.kernel_shape[0] + self.pads[0] + self.pads[2]) / self.strides[0] + 1;
    let output_w = (self.input_shape[2] - self.kernel_shape[1] + self.pads[1] + self.pads[3]) / self.strides[1] + 1;

    let mut candidates = vec![ArrayD::<Fr>::zeros(IxDyn(&[1, output_h * output_w, C_in])); self.kernel_shape[0] * self.kernel_shape[1]];
    for k in 0..self.kernel_shape[0] * self.kernel_shape[1] {
      for i in 0..output_h {
        for j in 0..output_w {
          let k_i = k / self.kernel_shape[1];
          let k_j = k % self.kernel_shape[1];
          let input_i = i * self.strides[0];
          let input_j = j * self.strides[1];
          for c in 0..C_in {
            // consider padding = 1, 1, 1, 1
            if (input_i + k_i) == 0 || (input_j + k_j) == 0 || (input_i + k_i) > self.input_shape[1] || (input_j + k_j) > self.input_shape[2] {
              candidates[k][[0, i * output_w + j, c]] = Fr::zero();
            } else {
              candidates[k][[0, i * output_w + j, c]] = input[[0, (input_i + k_i - 1) * self.input_shape[2] + input_j + k_j - 1, c]];
            }
          }
        }
      }
      candidates[k] = util::pad_to_pow_of_two(&candidates[k], &Fr::zero());
    }

    Ok(candidates)
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let C_o = self.input_shape[3];
    let output_h = (self.input_shape[1] - self.kernel_shape[0] + self.pads[0] + self.pads[2]) / self.strides[0] + 1;
    let output_w = (self.input_shape[2] - self.kernel_shape[1] + self.pads[1] + self.pads[3]) / self.strides[1] + 1;
    let data_zero = Data {
      raw: vec![Fr::zero(); C_o],
      poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
      r: Fr::zero(),
      g1: G1Projective::zero(),
    };
    let mut candidates =
      vec![ArrayD::from_shape_fn(IxDyn(&[1, output_h * output_w]), |_| data_zero.clone()); self.kernel_shape[0] * self.kernel_shape[1]];
    for k in 0..self.kernel_shape[0] * self.kernel_shape[1] {
      for i in 0..output_h {
        for j in 0..output_w {
          let k_i = k / self.kernel_shape[1];
          let k_j = k % self.kernel_shape[1];
          let input_i = i * self.strides[0];
          let input_j = j * self.strides[1];
          if (input_i + k_i) == 0 || (input_j + k_j) == 0 || (input_i + k_i) > self.input_shape[1] || (input_j + k_j) > self.input_shape[2] {
            candidates[k][[0, i * output_w + j]] = data_zero.clone();
          } else {
            candidates[k][[0, i * output_w + j]] = inputs[0][[0, (input_i + k_i - 1) * self.input_shape[2] + input_j + k_j - 1]].clone();
          }
        }
      }
      candidates[k] = util::pad_to_pow_of_two(&candidates[k], &data_zero);
    }
    candidates
  }
}

#[derive(Debug)]
pub struct MaxPool2dSelectorBasicBlock {
  pub input_shape: Vec<usize>,  // [1, H_in, W_in, C_in]
  pub kernel_shape: Vec<usize>, // [k_h, k_w]
  pub strides: Vec<usize>,      // [s_h, s_w]
  pub pads: Vec<usize>,         // [p_h, p_w, p_h, p_w]
}
impl BasicBlock for MaxPool2dSelectorBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let k = self.kernel_shape[0] * self.kernel_shape[1];
    assert!(inputs.len() == k);
    let C_in = self.input_shape[3];
    let output_h = (self.input_shape[1] - self.kernel_shape[0] + self.pads[0] + self.pads[2]) / self.strides[0] + 1;
    let output_w = (self.input_shape[2] - self.kernel_shape[1] + self.pads[1] + self.pads[3]) / self.strides[1] + 1;

    let mut selectors = vec![ArrayD::<Fr>::zeros(IxDyn(&[1, output_h * output_w, C_in])); self.kernel_shape[0] * self.kernel_shape[1]];
    for i in 0..output_h {
      for j in 0..output_w {
        for c in 0..C_in {
          let mut cq_max = inputs[0][[0, i * output_w + j, c]];
          let mut max_k = 0;
          for k in 0..self.kernel_shape[0] * self.kernel_shape[1] {
            let cq = inputs[k][[0, i * output_w + j, c]];
            let a: i128 = 1;
            if cq < Fr::from(a << 127) && cq >= cq_max {
              cq_max = cq;
              max_k = k;
            }
          }
          selectors[max_k][[0, i * output_w + j, c]] = Fr::one();
        }
      }
    }

    for i in 0..k {
      selectors[i] = util::pad_to_pow_of_two(&selectors[i], &Fr::zero());
    }

    Ok(selectors)
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
    let input_shape = input_shapes[0]; // [1, H_in * W_in, C_in]

    let C_in = input_shape[2];
    let ch_dims: Vec<usize> = match attributes.iter().filter(|x| x.name == "kernel_shape").next() {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => panic!("kernel_shape not found"),
    };

    // only support square image for now
    let input_h = (input_shapes[0][1] as f64).sqrt() as usize;
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

    // MaxPool2d
    let k = ch_dims[0] * ch_dims[1];
    let maxpool_candidate = graph.addBB(Box::new(MaxPool2dCandidateBasicBlock {
      input_shape: vec![1, dims[0], dims[1], C_in],
      kernel_shape: ch_dims.clone(),
      strides: strides.clone(),
      pads: pads.clone(),
    }));
    let maxpool_candidate_output = graph.addNode(maxpool_candidate, vec![(-1, 0)]);

    let maxpool_selector = graph.addBB(Box::new(MaxPool2dSelectorBasicBlock {
      input_shape: vec![1, dims[0], dims[1], C_in],
      kernel_shape: ch_dims.clone(),
      strides: strides.clone(),
      pads: pads.clone(),
    }));
    let maxpool_selector_output = graph.addNode(maxpool_selector, (0..k).map(|x| (maxpool_candidate_output, x)).collect());
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    // Check if the selector is all boolean
    for idx in 0..k {
      let _ = graph.addNode(bool_check, vec![(maxpool_selector_output, idx)]);
    }

    // Pointwise multiplication between the selector and the candidates
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {
        len: util::next_pow(C_in as u32) as usize,
      }),
      N: 1,
    }));

    let mut mul_outputs = Vec::new();
    for idx in 0..k {
      let mul_output = graph.addNode(mul, vec![(maxpool_selector_output, idx), (maxpool_candidate_output, idx)]);
      mul_outputs.push((mul_output, 0));
    }

    // Multiple add to sum the pointwise multiplications of the selector and the candidates
    let multi_add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MultipleAddBasicBlock {}),
      N: 1,
    }));
    let maxpool_output = graph.addNode(multi_add, mul_outputs);

    // Sub
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    // CQ to check if x >= 0
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: util::next_pow(C_in as u32) as usize,
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    for idx in 0..k {
      let sub_output = graph.addNode(sub, vec![(maxpool_output, 0), (maxpool_candidate_output, idx)]);
      // Check if the remainder sub_output is non-negative
      let _ = graph.addNode(range_check, vec![(sub_output, 0)]);
    }

    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));
    let mut ones = ArrayD::<Fr>::zeros(IxDyn(&[1, util::next_pow((C_in) as u32) as usize]));
    for i in 0..C_in {
      ones[[0, i]] = Fr::one();
    }
    let const_ones = graph.addBB(Box::new(Const2BasicBlock { c: ones }));
    let const_ones_output = graph.addNode(const_ones, vec![]);
    let add_output = graph.addNode(multi_add, (0..k).map(|x| (maxpool_selector_output, x)).collect());
    //let _ = graph.addNode(eq, vec![(add_output, 0), (const_ones_output, 0)]);

    let output_h = (dims[0] - ch_dims[0] + pads[0] + pads[2]) / strides[0] + 1;
    let output_w = (dims[1] - ch_dims[1] + pads[1] + pads[3]) / strides[1] + 1;
    let output_shape = vec![1, output_h * output_w, C_in];
    graph.outputs.push((maxpool_output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
