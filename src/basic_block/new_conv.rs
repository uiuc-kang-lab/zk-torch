use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{arr0, azip, s, ArrayD, Axis, IxDyn};
use rand::rngs::StdRng;
use rayon::prelude::*;

/*
  All basic blocks in this file are used to perform the customized addition of our
  special convolutional layers (where we put the kernel dimension in the last dimension).
  For instance, in the case of a 2D convolution, we can use CQLinBasicBlock to update the input tensor
  from shape [1, H_in * W_in, C_in] to shape [1, H_in * W_in, C_out].
  Then, we can use Conv2DAddBasicBlock here to perform the convolutional addition to map the tensor
  from [1, H_in * W_in, C_out] to [1, H_out * W_out, C_out].
*/

#[derive(Debug, Clone)]
pub struct ConvShapeHelper {
  pub d: Option<i32>,
  pub h: i32,
  pub w: i32,
  pub k_d: Option<i32>,
  pub k_h: i32,
  pub k_w: i32,
  pub s_d: Option<i32>,
  pub s_h: i32,
  pub s_w: i32,
  pub p_d_front: Option<i32>,
  pub p_h_top: i32,
  pub p_w_left: i32,
  pub p_d_back: Option<i32>,
  pub p_h_bottom: i32,
  pub p_w_right: i32,
  pub out_channels: usize,
}

impl From<&Conv3DTransposeBasicBlock> for ConvShapeHelper {
  fn from(conv: &Conv3DTransposeBasicBlock) -> Self {
    Self {
      d: Some(conv.input_shape[0]),
      h: conv.input_shape[1],
      w: conv.input_shape[2],
      k_d: Some(conv.kernel_shape[0]),
      k_h: conv.kernel_shape[1],
      k_w: conv.kernel_shape[2],
      s_d: Some(conv.stride[0]),
      s_h: conv.stride[1],
      s_w: conv.stride[2],
      p_d_front: Some(conv.padding[0]),
      p_h_top: conv.padding[1],
      p_w_left: conv.padding[2],
      p_d_back: Some(conv.padding[3]),
      p_h_bottom: conv.padding[4],
      p_w_right: conv.padding[5],
      out_channels: conv.out_channels,
    }
  }
}

impl From<&Conv3DAddBasicBlock> for ConvShapeHelper {
  fn from(conv: &Conv3DAddBasicBlock) -> Self {
    Self {
      d: Some(conv.input_shape[0]),
      h: conv.input_shape[1],
      w: conv.input_shape[2],
      k_d: Some(conv.kernel_shape[0]),
      k_h: conv.kernel_shape[1],
      k_w: conv.kernel_shape[2],
      s_d: Some(conv.stride[0]),
      s_h: conv.stride[1],
      s_w: conv.stride[2],
      p_d_front: Some(conv.padding[0]),
      p_h_top: conv.padding[1],
      p_w_left: conv.padding[2],
      p_d_back: Some(conv.padding[3]),
      p_h_bottom: conv.padding[4],
      p_w_right: conv.padding[5],
      out_channels: conv.out_channels,
    }
  }
}

impl From<&Conv2DAddBasicBlock> for ConvShapeHelper {
  fn from(conv: &Conv2DAddBasicBlock) -> Self {
    Self {
      d: None,
      h: conv.input_shape[0],
      w: conv.input_shape[1],
      k_d: None,
      k_h: conv.kernel_shape[0],
      k_w: conv.kernel_shape[1],
      s_d: None,
      s_h: conv.stride[0],
      s_w: conv.stride[1],
      p_d_front: None,
      p_h_top: conv.padding[0],
      p_w_left: conv.padding[1],
      p_d_back: None,
      p_h_bottom: conv.padding[2],
      p_w_right: conv.padding[3],
      out_channels: conv.out_channels,
    }
  }
}

#[derive(Debug)]
pub struct Conv3DTransposeBasicBlock {
  pub input_shape: Vec<i32>,  // [D, H, W]
  pub kernel_shape: Vec<i32>, // [k_d, k_h, k_w]
  pub stride: Vec<i32>,       // [s_d, s_h, s_w]
  pub padding: Vec<i32>,      // [p_d_front, p_h_top, p_w_left, p_d_back, p_h_bottom, p_w_right]
  pub out_channels: usize,
}

// Conv3DAddBasicBlock is a basic block that performs convolution and addition.
// Inputs: d*h*w tensors, each with shape [1, D*H*W, C_o]
// Outputs: 1 tensor with shape [1, D_o*H_o*W_o, C_o]
impl BasicBlock for Conv3DTransposeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let shape = ConvShapeHelper::from(self);
    assert!(shape.k_d == shape.s_d && shape.k_h == shape.s_h && shape.k_w == shape.s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(shape.p_d_front == Some(0) && shape.p_h_top == 0 && shape.p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(shape.p_d_back == Some(0) && shape.p_h_bottom == 0 && shape.p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (shape.d.unwrap() - 1) * shape.s_d.unwrap() - shape.p_d_front.unwrap() - shape.p_d_back.unwrap() + (shape.k_d.unwrap() - 1) + 1;
    let H_o = (shape.h - 1) * shape.s_h - shape.p_h_top - shape.p_h_bottom + (shape.k_h - 1) + 1;
    let W_o = (shape.w - 1) * shape.s_w - shape.p_w_left - shape.p_w_right + (shape.k_w - 1) + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, shape.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let x_in = i / shape.s_d.unwrap();
            let y_in = j / shape.s_h;
            let z_in = k / shape.s_w;
            let x = i % shape.s_d.unwrap();
            let y = j % shape.s_h;
            let z = k % shape.s_w;
            let input_idx = x * shape.k_h * shape.k_w + y * shape.k_w + z;
            let input_idx = input_idx as usize;
            if x_in >= 0 && x_in < shape.d.unwrap() && y_in >= 0 && y_in < shape.h && z_in >= 0 && z_in < shape.w {
              let x_in = x_in as usize;
              let y_in = y_in as usize;
              let z_in = z_in as usize;
              let i = i as usize;
              let j = j as usize;
              let k = k as usize;
              channel[[0, i * (H_o as usize) * (W_o as usize) + j * (W_o as usize) + k]] =
                inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in, c]];
            }
          }
        }
      }
    });
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let shape = ConvShapeHelper::from(self);
    assert!(shape.k_d.unwrap() == shape.s_d.unwrap() && shape.k_h == shape.s_h && shape.k_w == shape.s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(shape.p_d_front == Some(0) && shape.p_h_top == 0 && shape.p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(shape.p_d_back == Some(0) && shape.p_h_bottom == 0 && shape.p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (shape.d.unwrap() - 1) * shape.s_d.unwrap() - shape.p_d_front.unwrap() - shape.p_d_back.unwrap() + (shape.k_d.unwrap() - 1) + 1;
    let H_o = (shape.h - 1) * shape.s_h - shape.p_h_top - shape.p_h_bottom + (shape.k_h - 1) + 1;
    let W_o = (shape.w - 1) * shape.s_w - shape.p_w_left - shape.p_w_right + (shape.k_w - 1) + 1;
    let data_list: Vec<Data> = (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let mut data = Data {
              raw: outputs[0].slice(s![0, i * H_o * W_o + j * W_o + k, ..]).clone().to_vec(),
              poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
              r: Fr::zero(),
              g1: G1Projective::zero(),
            };

            let x_in = i / shape.s_d.unwrap();
            let y_in = j / shape.s_h;
            let z_in = k / shape.s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % shape.s_d.unwrap();
            let y = j % shape.s_h;
            let z = k % shape.s_w;
            let input_idx = x * shape.k_h * shape.k_w + y * shape.k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in]];
            data.poly += &input.poly;
            data.g1 += input.g1;
            data.r += input.r;

            data
          })
          .collect::<Vec<_>>()
      })
      .collect();
    let D_o = D_o as usize;
    let H_o = H_o as usize;
    let W_o = W_o as usize;
    let output = ArrayD::from_shape_vec(IxDyn(&[1, D_o * H_o * W_o]), data_list).unwrap();
    let output = util::pad_to_pow_of_two(
      &output,
      &Data {
        raw: vec![Fr::zero(); shape.out_channels],
        poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
        r: Fr::zero(),
        g1: G1Projective::zero(),
      },
    );
    vec![output]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let shape = ConvShapeHelper::from(self);
    assert!(shape.k_d.unwrap() == shape.s_d.unwrap() && shape.k_h == shape.s_h && shape.k_w == shape.s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(shape.p_d_front == Some(0) && shape.p_h_top == 0 && shape.p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(shape.p_d_back == Some(0) && shape.p_h_bottom == 0 && shape.p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (shape.d.unwrap() - 1) * shape.s_d.unwrap() - shape.p_d_front.unwrap() - shape.p_d_back.unwrap() + (shape.k_d.unwrap() - 1) + 1;
    let H_o = (shape.h - 1) * shape.s_h - shape.p_h_top - shape.p_h_bottom + (shape.k_h - 1) + 1;
    let W_o = (shape.w - 1) * shape.s_w - shape.p_w_left - shape.p_w_right + (shape.k_w - 1) + 1;
    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let x_in = i / shape.s_d.unwrap();
            let y_in = j / shape.s_h;
            let z_in = k / shape.s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % shape.s_d.unwrap();
            let y = j % shape.s_h;
            let z = k % shape.s_w;
            let input_idx = x * shape.k_h * shape.k_w + y * shape.k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in]];
            let i = i as usize;
            let j = j as usize;
            let k = k as usize;
            assert!(input.g1 == outputs[0][[0, i * ((H_o * W_o) as usize) + j * (W_o as usize) + k]].g1);
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    vec![]
  }
}

#[derive(Debug)]
pub struct Conv3DAddBasicBlock {
  pub input_shape: Vec<i32>,  // [D, H, W]
  pub kernel_shape: Vec<i32>, // [k_d, k_h, k_w]
  pub stride: Vec<i32>,       // [s_d, s_h, s_w]
  pub padding: Vec<i32>,      // [p_d_front, p_h_top, p_w_left, p_d_back, p_h_bottom, p_w_right]
  pub out_channels: usize,
}

// Conv3DAddBasicBlock is a basic block that performs convolution and addition.
// Inputs: d*h*w tensors, each with shape [1, D*H*W, C_o]
// Outputs: 1 tensor with shape [1, D_o*H_o*W_o, C_o]
impl BasicBlock for Conv3DAddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let shape = ConvShapeHelper::from(self);
    let D_o = (shape.d.unwrap() - shape.k_d.unwrap() + shape.p_d_front.unwrap() + shape.p_d_back.unwrap()) / shape.s_d.unwrap() + 1;
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, shape.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let mut sum = Fr::zero();
            for x in 0..shape.k_d.unwrap() {
              for y in 0..shape.k_h {
                for z in 0..shape.k_w {
                  let x_in = i * shape.s_d.unwrap() + x - shape.p_d_front.unwrap();
                  let y_in = j * shape.s_h + y - shape.p_h_top;
                  let z_in = k * shape.s_w + z - shape.p_w_left;
                  let input_idx = x * shape.k_h * shape.k_w + y * shape.k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < shape.d.unwrap() && y_in >= 0 && y_in < shape.h && z_in >= 0 && z_in < shape.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in, c]];
                  }
                }
              }
            }
            let i = i as usize;
            let j = j as usize;
            let k = k as usize;
            channel[[0, i * (H_o as usize) * (W_o as usize) + j * (W_o as usize) + k]] = sum;
          }
        }
      }
    });
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let shape = ConvShapeHelper::from(self);
    let D_o = (shape.d.unwrap() - shape.k_d.unwrap() + shape.p_d_front.unwrap() + shape.p_d_back.unwrap()) / shape.s_d.unwrap() + 1;
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;
    let data_list: Vec<Data> = (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let mut data = Data {
              raw: outputs[0].slice(s![0, i * H_o * W_o + j * W_o + k, ..]).clone().to_vec(),
              poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
              r: Fr::zero(),
              g1: G1Projective::zero(),
            };

            for x in 0..shape.k_d.unwrap() {
              for y in 0..shape.k_h {
                for z in 0..shape.k_w {
                  let x_in = i * shape.s_d.unwrap() + x - shape.p_d_front.unwrap();
                  let y_in = j * shape.s_h + y - shape.p_h_top;
                  let z_in = k * shape.s_w + z - shape.p_w_left;
                  let input_idx = x * shape.k_h * shape.k_w + y * shape.k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < shape.d.unwrap() && y_in >= 0 && y_in < shape.h && z_in >= 0 && z_in < shape.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    let input = &inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in]];
                    data.poly += &input.poly;
                    data.g1 += input.g1;
                    data.r += input.r;
                  }
                }
              }
            }
            data
          })
          .collect::<Vec<_>>()
      })
      .collect();
    let D_o = D_o as usize;
    let H_o = H_o as usize;
    let W_o = W_o as usize;
    let output = ArrayD::from_shape_vec(IxDyn(&[1, D_o * H_o * W_o]), data_list).unwrap();
    let output = util::pad_to_pow_of_two(
      &output,
      &Data {
        raw: vec![Fr::zero(); shape.out_channels],
        poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
        r: Fr::zero(),
        g1: G1Projective::zero(),
      },
    );
    vec![output]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let shape = ConvShapeHelper::from(self);
    let D_o = (shape.d.unwrap() - shape.k_d.unwrap() + shape.p_d_front.unwrap() + shape.p_d_back.unwrap()) / shape.s_d.unwrap() + 1;
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;

    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let mut sum = G1Projective::zero();
            for x in 0..shape.k_d.unwrap() {
              for y in 0..shape.k_h {
                for z in 0..shape.k_w {
                  let x_in = i * shape.s_d.unwrap() + x - shape.p_d_front.unwrap();
                  let y_in = j * shape.s_h + y - shape.p_h_top;
                  let z_in = k * shape.s_w + z - shape.p_w_left;
                  let input_idx = (x * shape.k_h * shape.k_w + y * shape.k_w + z) as usize;
                  if x_in >= 0 && x_in < shape.d.unwrap() && y_in >= 0 && y_in < shape.h && z_in >= 0 && z_in < shape.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += &inputs[input_idx][[0, x_in * (shape.h as usize) * (shape.w as usize) + y_in * (shape.w as usize) + z_in]].g1;
                  }
                }
              }
            }
            let i = i as usize;
            let j = j as usize;
            let k = k as usize;
            assert!(sum == outputs[0][[0, i * ((H_o * W_o) as usize) + j * (W_o as usize) + k]].g1);
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    vec![]
  }
}

#[derive(Debug)]
pub struct Conv2DAddBasicBlock {
  pub input_shape: Vec<i32>,  // [H, W]
  pub kernel_shape: Vec<i32>, // [k_h, k_w]
  pub stride: Vec<i32>,       // [s_h, s_w]
  pub padding: Vec<i32>,      // [p_h_top, p_w_left, p_h_bottom, p_w_right]
  pub out_channels: usize,
}

// Conv2DAddBasicBlock is a basic block that performs convolution and addition.
// Inputs: h*w tensors, each with shape [1, H*W, C_o]
// Outputs: 1 tensor with shape [1, H_o*W_o, C_o]
impl BasicBlock for Conv2DAddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let shape = ConvShapeHelper::from(self);
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (H_o * W_o) as usize, shape.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..H_o {
        for j in 0..W_o {
          let mut sum = Fr::zero();
          for x in 0..shape.k_h {
            for y in 0..shape.k_w {
              let x_in = i * shape.s_h + x - shape.p_h_top;
              let y_in = j * shape.s_w + y - shape.p_w_left;
              let input_idx = x * shape.k_w + y;
              let input_idx = input_idx as usize;
              if x_in >= 0 && x_in < shape.h && y_in >= 0 && y_in < shape.w {
                let x_in = x_in as usize;
                let y_in = y_in as usize;
                sum += inputs[input_idx][[0, x_in * (shape.w as usize) + y_in, c]];
              }
            }
          }
          let i = i as usize;
          let j = j as usize;
          channel[[0, i * (W_o as usize) + j]] = sum;
        }
      }
    });
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let shape = ConvShapeHelper::from(self);
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;
    let data_list: Vec<Data> = (0..H_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..W_o)
          .map(move |j| {
            let mut data = Data {
              raw: outputs[0].slice(s![0, i * W_o + j, ..]).clone().to_vec(),
              poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
              r: Fr::zero(),
              g1: G1Projective::zero(),
            };

            for x in 0..shape.k_h {
              for y in 0..shape.k_w {
                let x_in = i * shape.s_h + x - shape.p_h_top;
                let y_in = j * shape.s_w + y - shape.p_w_left;
                let input_idx = x * shape.k_w + y;

                let input_idx = input_idx as usize;

                if x_in >= 0 && x_in < shape.h && y_in >= 0 && y_in < shape.w {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  let input = &inputs[input_idx][[0, x_in * (shape.w as usize) + y_in]];
                  data.poly += &input.poly;
                  data.g1 += input.g1;
                  data.r += input.r;
                }
              }
            }
            data
          })
          .collect::<Vec<_>>()
      })
      .collect();
    let H_o = H_o as usize;
    let W_o = W_o as usize;
    let output = ArrayD::from_shape_vec(IxDyn(&[1, H_o * W_o]), data_list).unwrap();
    let output = util::pad_to_pow_of_two(
      &output,
      &Data {
        raw: vec![Fr::zero(); shape.out_channels],
        poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
        r: Fr::zero(),
        g1: G1Projective::zero(),
      },
    );
    vec![output]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let shape = ConvShapeHelper::from(self);
    let H_o = (shape.h - shape.k_h + shape.p_h_top + shape.p_h_bottom) / shape.s_h + 1;
    let W_o = (shape.w - shape.k_w + shape.p_w_left + shape.p_w_right) / shape.s_w + 1;
    (0..H_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..W_o)
          .map(move |j| {
            let mut sum = G1Projective::zero();
            for x in 0..shape.k_h {
              for y in 0..shape.k_w {
                let x_in = i * shape.s_h + x - shape.p_h_top;
                let y_in = j * shape.s_w + y - shape.p_w_left;
                let input_idx = (x * shape.k_w + y) as usize;
                if x_in >= 0 && x_in < shape.h && y_in >= 0 && y_in < shape.w {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  sum += &inputs[input_idx][[0, x_in * (shape.w as usize) + y_in]].g1;
                }
              }
            }
            let i = i as usize;
            let j = j as usize;
            assert!(sum == outputs[0][[0, i * (W_o as usize) + j]].g1);
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    vec![]
  }
}

// This basic block is only used in RetinaNet where we need to concatenate the outputs of multiple conv heads along axis 1.
#[derive(Debug)]
pub struct MultiHeadConv2dAggBasicBlock {
  pub input_shape: Vec<usize>, // [1, H_out * W_out, head_dim]
}
impl BasicBlock for MultiHeadConv2dAggBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let head_num = inputs.len();
    let mut final_output_shape = self.input_shape.clone();
    final_output_shape[1] *= head_num;

    let mut result = ArrayD::<Fr>::zeros(IxDyn(&final_output_shape));
    for head in 0..head_num {
      let input_slice = util::slice_nd_array(inputs[head].clone(), &self.input_shape);
      result.slice_axis_mut(Axis(1), (head * self.input_shape[1]..(head + 1) * self.input_shape[1]).into()).assign(&input_slice);
    }
    result = util::pad_to_pow_of_two(&result, &Fr::zero());

    Ok(vec![result])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let head_num = inputs.len();
    let mut final_output_shape = vec![self.input_shape[0], self.input_shape[1]];
    final_output_shape[1] *= head_num;
    let data_zero = Data {
      raw: vec![Fr::zero(); self.input_shape[2]],
      poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
      r: Fr::zero(),
      g1: G1Projective::zero(),
    };
    let mut result = ArrayD::from_shape_fn(IxDyn(&final_output_shape), |_| data_zero.clone());
    for head in 0..head_num {
      let input_slice = util::slice_nd_array(inputs[head].clone(), &[self.input_shape[0], self.input_shape[1]]);
      result.slice_axis_mut(Axis(1), (head * self.input_shape[1]..(head + 1) * self.input_shape[1]).into()).assign(&input_slice);
    }
    result = util::pad_to_pow_of_two(&result, &data_zero);
    vec![result]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let head_num = inputs.len();
    let output = outputs[0];
    for head in 0..head_num {
      for i in 0..self.input_shape[1] {
        assert!(output[[0, head * self.input_shape[1] + i]].g1 == inputs[head][[0, i]].g1);
      }
    }
    vec![]
  }
}
