/*
  All basic blocks in this file are used to perform the customized addition of our
  special convolutional layers (where we put the kernel dimension in the last dimension).
  For instance, in the case of a 2D convolution, we can use CQLinBasicBlock to update the input tensor
  from shape [1, H_in * W_in, C_in] to shape [1, H_in * W_in, C_out].
  Then, we can use Conv2DAddBasicBlock here to perform the convolutional addition to map the tensor
  from [1, H_in * W_in, C_out] to [1, H_out * W_out, C_out].
*/
use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{arr0, azip, s, ArrayD, Axis, IxDyn};
use rand::rngs::StdRng;
use rayon::prelude::*;

// Conv3DTransposeBasicBlock is a basic block that performs convolution transpose and addition.
// (Please refer to https://onnx.ai/onnx/operators/onnx__ConvTranspose.html for the definition of convolution transpose.)
// Inputs: d*h*w tensors, each with shape [1, D*H*W, C_o]
// Outputs: 1 tensor with shape [1, D_o*H_o*W_o, C_o]
// Note this basicblock is only supported for the case where
// (1) stride is equal to kernel size.
// (2) padding is zero.
#[derive(Debug)]
pub struct Conv3DTransposeBasicBlock {
  // input size (d, h, w)
  pub d: i32,
  pub h: i32,
  pub w: i32,
  // kernel size (k_d, k_h, k_w)
  pub k_d: i32,
  pub k_h: i32,
  pub k_w: i32,
  // stride (s_d, s_h, s_w)
  pub s_d: i32,
  pub s_h: i32,
  pub s_w: i32,
  // padding (p_d_front, p_h_top, p_w_left, p_d_back, p_h_bottom, p_w_right)
  pub p_d_front: i32,
  pub p_h_top: i32,
  pub p_w_left: i32,
  pub p_d_back: i32,
  pub p_h_bottom: i32,
  pub p_w_right: i32,
  // output channels
  pub out_channels: usize,
}

impl BasicBlock for Conv3DTransposeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(self.k_d == self.s_d && self.k_h == self.s_h && self.k_w == self.s_w);
    assert!(self.p_d_front == 0 && self.p_h_top == 0 && self.p_w_left == 0);
    assert!(self.p_d_back == 0 && self.p_h_bottom == 0 && self.p_w_right == 0);
    let D_o = (self.d - 1) * self.s_d - self.p_d_front - self.p_d_back + (self.k_d - 1) + 1;
    let H_o = (self.h - 1) * self.s_h - self.p_h_top - self.p_h_bottom + (self.k_h - 1) + 1;
    let W_o = (self.w - 1) * self.s_w - self.p_w_left - self.p_w_right + (self.k_w - 1) + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, self.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let x_in = i / self.s_d;
            let y_in = j / self.s_h;
            let z_in = k / self.s_w;
            let x = i % self.s_d;
            let y = j % self.s_h;
            let z = k % self.s_w;
            let input_idx = x * self.k_h * self.k_w + y * self.k_w + z;
            let input_idx = input_idx as usize;
            if x_in >= 0 && x_in < self.d && y_in >= 0 && y_in < self.h && z_in >= 0 && z_in < self.w {
              let x_in = x_in as usize;
              let y_in = y_in as usize;
              let z_in = z_in as usize;
              let i = i as usize;
              let j = j as usize;
              let k = k as usize;
              channel[[0, i * (H_o as usize) * (W_o as usize) + j * (W_o as usize) + k]] =
                inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in, c]];
            }
          }
        }
      }
    });
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    assert!(self.k_d == self.s_d && self.k_h == self.s_h && self.k_w == self.s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(self.p_d_front == 0 && self.p_h_top == 0 && self.p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(self.p_d_back == 0 && self.p_h_bottom == 0 && self.p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (self.d - 1) * self.s_d - self.p_d_front - self.p_d_back + (self.k_d - 1) + 1;
    let H_o = (self.h - 1) * self.s_h - self.p_h_top - self.p_h_bottom + (self.k_h - 1) + 1;
    let W_o = (self.w - 1) * self.s_w - self.p_w_left - self.p_w_right + (self.k_w - 1) + 1;
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

            let x_in = i / self.s_d;
            let y_in = j / self.s_h;
            let z_in = k / self.s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % self.s_d;
            let y = j % self.s_h;
            let z = k % self.s_w;
            let input_idx = x * self.k_h * self.k_w + y * self.k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in]];
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
        raw: vec![Fr::zero(); self.out_channels],
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
    assert!(self.k_d == self.s_d && self.k_h == self.s_h && self.k_w == self.s_w);
    assert!(self.p_d_front == 0 && self.p_h_top == 0 && self.p_w_left == 0);
    assert!(self.p_d_back == 0 && self.p_h_bottom == 0 && self.p_w_right == 0);
    let D_o = (self.d - 1) * self.s_d - self.p_d_front - self.p_d_back + (self.k_d - 1) + 1;
    let H_o = (self.h - 1) * self.s_h - self.p_h_top - self.p_h_bottom + (self.k_h - 1) + 1;
    let W_o = (self.w - 1) * self.s_w - self.p_w_left - self.p_w_right + (self.k_w - 1) + 1;
    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let x_in = i / self.s_d;
            let y_in = j / self.s_h;
            let z_in = k / self.s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % self.s_d;
            let y = j % self.s_h;
            let z = k % self.s_w;
            let input_idx = x * self.k_h * self.k_w + y * self.k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in]];
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
  // input size (d, h, w)
  pub d: i32,
  pub h: i32,
  pub w: i32,
  // kernel size (k_d, k_h, k_w)
  pub k_d: i32,
  pub k_h: i32,
  pub k_w: i32,
  // stride (s_d, s_h, s_w)
  pub s_d: i32,
  pub s_h: i32,
  pub s_w: i32,
  // padding (p_d_front, p_h_top, p_w_left, p_d_back, p_h_bottom, p_w_right)
  pub p_d_front: i32,
  pub p_h_top: i32,
  pub p_w_left: i32,
  pub p_d_back: i32,
  pub p_h_bottom: i32,
  pub p_w_right: i32,
  pub out_channels: usize,
}

// Conv3DAddBasicBlock is a basic block that performs convolution and addition.
// Inputs: d*h*w tensors, each with shape [1, D*H*W, C_o]
// Outputs: 1 tensor with shape [1, D_o*H_o*W_o, C_o]
impl BasicBlock for Conv3DAddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let D_o = (self.d - self.k_d + self.p_d_front + self.p_d_back) / self.s_d + 1;
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, self.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let mut sum = Fr::zero();
            for x in 0..self.k_d {
              for y in 0..self.k_h {
                for z in 0..self.k_w {
                  let x_in = i * self.s_d + x - self.p_d_front;
                  let y_in = j * self.s_h + y - self.p_h_top;
                  let z_in = k * self.s_w + z - self.p_w_left;
                  let input_idx = x * self.k_h * self.k_w + y * self.k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < self.d && y_in >= 0 && y_in < self.h && z_in >= 0 && z_in < self.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in, c]];
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
    let D_o = (self.d - self.k_d + self.p_d_front + self.p_d_back) / self.s_d + 1;
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;
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

            for x in 0..self.k_d {
              for y in 0..self.k_h {
                for z in 0..self.k_w {
                  let x_in = i * self.s_d + x - self.p_d_front;
                  let y_in = j * self.s_h + y - self.p_h_top;
                  let z_in = k * self.s_w + z - self.p_w_left;
                  let input_idx = x * self.k_h * self.k_w + y * self.k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < self.d && y_in >= 0 && y_in < self.h && z_in >= 0 && z_in < self.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    let input = &inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in]];
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
        raw: vec![Fr::zero(); self.out_channels],
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
    let D_o = (self.d - self.k_d + self.p_d_front + self.p_d_back) / self.s_d + 1;
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;

    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let mut sum = G1Projective::zero();
            for x in 0..self.k_d {
              for y in 0..self.k_h {
                for z in 0..self.k_w {
                  let x_in = i * self.s_d + x - self.p_d_front;
                  let y_in = j * self.s_h + y - self.p_h_top;
                  let z_in = k * self.s_w + z - self.p_w_left;
                  let input_idx = (x * self.k_h * self.k_w + y * self.k_w + z) as usize;
                  if x_in >= 0 && x_in < self.d && y_in >= 0 && y_in < self.h && z_in >= 0 && z_in < self.w {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += &inputs[input_idx][[0, x_in * (self.h as usize) * (self.w as usize) + y_in * (self.w as usize) + z_in]].g1;
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
  // input size (h, w)
  pub h: i32,
  pub w: i32,
  // kernel size (k_h, k_w)
  pub k_h: i32,
  pub k_w: i32,
  // stride (s_h, s_w)
  pub s_h: i32,
  pub s_w: i32,
  // padding (p_h_top, p_w_left, p_h_bottom, p_w_right)
  pub p_h_top: i32,
  pub p_w_left: i32,
  pub p_h_bottom: i32,
  pub p_w_right: i32,
  // output channels
  pub out_channels: usize,
}

// Conv2DAddBasicBlock is a basic block that performs convolution and addition.
// Inputs: h*w tensors, each with shape [1, H*W, C_o]
// Outputs: 1 tensor with shape [1, H_o*W_o, C_o]
impl BasicBlock for Conv2DAddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (H_o * W_o) as usize, self.out_channels]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..H_o {
        for j in 0..W_o {
          let mut sum = Fr::zero();
          for x in 0..self.k_h {
            for y in 0..self.k_w {
              let x_in = i * self.s_h + x - self.p_h_top;
              let y_in = j * self.s_w + y - self.p_w_left;
              let input_idx = x * self.k_w + y;
              let input_idx = input_idx as usize;
              if x_in >= 0 && x_in < self.h && y_in >= 0 && y_in < self.w {
                let x_in = x_in as usize;
                let y_in = y_in as usize;
                sum += inputs[input_idx][[0, x_in * (self.w as usize) + y_in, c]];
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
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;
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

            for x in 0..self.k_h {
              for y in 0..self.k_w {
                let x_in = i * self.s_h + x - self.p_h_top;
                let y_in = j * self.s_w + y - self.p_w_left;
                let input_idx = x * self.k_w + y;

                let input_idx = input_idx as usize;

                if x_in >= 0 && x_in < self.h && y_in >= 0 && y_in < self.w {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  let input = &inputs[input_idx][[0, x_in * (self.w as usize) + y_in]];
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
        raw: vec![Fr::zero(); self.out_channels],
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
    let H_o = (self.h - self.k_h + self.p_h_top + self.p_h_bottom) / self.s_h + 1;
    let W_o = (self.w - self.k_w + self.p_w_left + self.p_w_right) / self.s_w + 1;
    (0..H_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..W_o)
          .map(move |j| {
            let mut sum = G1Projective::zero();
            for x in 0..self.k_h {
              for y in 0..self.k_w {
                let x_in = i * self.s_h + x - self.p_h_top;
                let y_in = j * self.s_w + y - self.p_w_left;
                let input_idx = (x * self.k_w + y) as usize;
                if x_in >= 0 && x_in < self.h && y_in >= 0 && y_in < self.w {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  sum += &inputs[input_idx][[0, x_in * (self.w as usize) + y_in]].g1;
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
