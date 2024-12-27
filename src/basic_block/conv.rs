use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{arr0, azip, s, ArrayD, Axis, IxDyn};
use rand::rngs::StdRng;
use rayon::prelude::*;

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
    let C_o = self.out_channels;
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    assert!(k_d == s_d && k_h == s_h && k_w == s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(p_d_front == 0 && p_h_top == 0 && p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(p_d_back == 0 && p_h_bottom == 0 && p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (D - 1) * s_d - p_d_front - p_d_back + (k_d - 1) + 1;
    let H_o = (H - 1) * s_h - p_h_top - p_h_bottom + (k_h - 1) + 1;
    let W_o = (W - 1) * s_w - p_w_left - p_w_right + (k_w - 1) + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, C_o]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let x_in = i / s_d;
            let y_in = j / s_h;
            let z_in = k / s_w;
            let x = i % s_d;
            let y = j % s_h;
            let z = k % s_w;
            let input_idx = x * k_h * k_w + y * k_w + z;
            let input_idx = input_idx as usize;
            if x_in >= 0 && x_in < D && y_in >= 0 && y_in < H && z_in >= 0 && z_in < W {
              let x_in = x_in as usize;
              let y_in = y_in as usize;
              let z_in = z_in as usize;
              let i = i as usize;
              let j = j as usize;
              let k = k as usize;
              channel[[0, i * (H_o as usize) * (W_o as usize) + j * (W_o as usize) + k]] =
                inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in, c]];
            }
          }
        }
      }
    });
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let C_o = self.out_channels;
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    assert!(k_d == s_d && k_h == s_h && k_w == s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(p_d_front == 0 && p_h_top == 0 && p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(p_d_back == 0 && p_h_bottom == 0 && p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (D - 1) * s_d - p_d_front - p_d_back + (k_d - 1) + 1;
    let H_o = (H - 1) * s_h - p_h_top - p_h_bottom + (k_h - 1) + 1;
    let W_o = (W - 1) * s_w - p_w_left - p_w_right + (k_w - 1) + 1;
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

            let x_in = i / s_d;
            let y_in = j / s_h;
            let z_in = k / s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % s_d;
            let y = j % s_h;
            let z = k % s_w;
            let input_idx = x * k_h * k_w + y * k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in]];
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
        raw: vec![Fr::zero(); C_o],
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
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    assert!(k_d == s_d && k_h == s_h && k_w == s_w); // The case where stride is equal to kernel size is not implemented for now.
    assert!(p_d_front == 0 && p_h_top == 0 && p_w_left == 0); // The case where padding is not zero is not implemented for now.
    assert!(p_d_back == 0 && p_h_bottom == 0 && p_w_right == 0); // The case where padding is not zero is not implemented for now.
    let D_o = (D - 1) * s_d - p_d_front - p_d_back + (k_d - 1) + 1;
    let H_o = (H - 1) * s_h - p_h_top - p_h_bottom + (k_h - 1) + 1;
    let W_o = (W - 1) * s_w - p_w_left - p_w_right + (k_w - 1) + 1;
    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let x_in = i / s_d;
            let y_in = j / s_h;
            let z_in = k / s_w;
            let x_in = x_in as usize;
            let y_in = y_in as usize;
            let z_in = z_in as usize;
            let x = i % s_d;
            let y = j % s_h;
            let z = k % s_w;
            let input_idx = x * k_h * k_w + y * k_w + z;
            let input_idx = input_idx as usize;
            let input = &inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in]];
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
    let C_o = self.out_channels;
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    let D_o = (D - k_d + p_d_front + p_d_back) / s_d + 1;
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (D_o * H_o * W_o) as usize, C_o]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..D_o {
        for j in 0..H_o {
          for k in 0..W_o {
            let mut sum = Fr::zero();
            for x in 0..k_d {
              for y in 0..k_h {
                for z in 0..k_w {
                  let x_in = i * s_d + x - p_d_front;
                  let y_in = j * s_h + y - p_h_top;
                  let z_in = k * s_w + z - p_w_left;
                  let input_idx = x * k_h * k_w + y * k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < D && y_in >= 0 && y_in < H && z_in >= 0 && z_in < W {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in, c]];
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
    let C_o = self.out_channels;
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    let D_o = (D - k_d + p_d_front + p_d_back) / s_d + 1;
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
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

            for x in 0..k_d {
              for y in 0..k_h {
                for z in 0..k_w {
                  let x_in = i * s_d + x - p_d_front;
                  let y_in = j * s_h + y - p_h_top;
                  let z_in = k * s_w + z - p_w_left;
                  let input_idx = x * k_h * k_w + y * k_w + z;
                  let input_idx = input_idx as usize;
                  if x_in >= 0 && x_in < D && y_in >= 0 && y_in < H && z_in >= 0 && z_in < W {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    let input = &inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in]];
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
        raw: vec![Fr::zero(); C_o],
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
    let D = self.input_shape[0];
    let H = self.input_shape[1];
    let W = self.input_shape[2];

    let k_d = self.kernel_shape[0];
    let k_h = self.kernel_shape[1];
    let k_w = self.kernel_shape[2];
    let s_d = self.stride[0];
    let s_h = self.stride[1];
    let s_w = self.stride[2];
    let p_d_front = self.padding[0];
    let p_h_top = self.padding[1];
    let p_w_left = self.padding[2];
    let p_d_back = self.padding[3];
    let p_h_bottom = self.padding[4];
    let p_w_right = self.padding[5];
    let D_o = (D - k_d + p_d_front + p_d_back) / s_d + 1;
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
    (0..D_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..H_o * W_o)
          .map(move |ind| {
            let j = ind / W_o;
            let k = ind % W_o;

            let mut sum = G1Projective::zero();
            for x in 0..k_d {
              for y in 0..k_h {
                for z in 0..k_w {
                  let x_in = i * s_d + x - p_d_front;
                  let y_in = j * s_h + y - p_h_top;
                  let z_in = k * s_w + z - p_w_left;
                  let input_idx = (x * k_h * k_w + y * k_w + z) as usize;
                  if x_in >= 0 && x_in < D && y_in >= 0 && y_in < H && z_in >= 0 && z_in < W {
                    let x_in = x_in as usize;
                    let y_in = y_in as usize;
                    let z_in = z_in as usize;
                    sum += &inputs[input_idx][[0, x_in * (H as usize) * (W as usize) + y_in * (W as usize) + z_in]].g1;
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
    let H = self.input_shape[0];
    let W = self.input_shape[1];
    let C_o = self.out_channels;
    let k_h = self.kernel_shape[0];
    let k_w = self.kernel_shape[1];
    let s_h = self.stride[0];
    let s_w = self.stride[1];
    let p_h_top = self.padding[0];
    let p_w_left = self.padding[1];
    let p_h_bottom = self.padding[2];
    let p_w_right = self.padding[3];
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
    let mut r = ArrayD::zeros(IxDyn(&[1, (H_o * W_o) as usize, C_o]));
    r.axis_iter_mut(Axis(2)).enumerate().par_bridge().for_each(|(c, mut channel)| {
      for i in 0..H_o {
        for j in 0..W_o {
          let mut sum = Fr::zero();
          for x in 0..k_h {
            for y in 0..k_w {
              let x_in = i * s_h + x - p_h_top;
              let y_in = j * s_w + y - p_w_left;
              let input_idx = x * k_w + y;
              let input_idx = input_idx as usize;
              if x_in >= 0 && x_in < H && y_in >= 0 && y_in < W {
                let x_in = x_in as usize;
                let y_in = y_in as usize;
                sum += inputs[input_idx][[0, x_in * (W as usize) + y_in, c]];
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
    let H = self.input_shape[0];
    let W = self.input_shape[1];
    let C_o = self.out_channels;
    let k_h = self.kernel_shape[0];
    let k_w = self.kernel_shape[1];
    let s_h = self.stride[0];
    let s_w = self.stride[1];
    let p_h_top = self.padding[0];
    let p_w_left = self.padding[1];
    let p_h_bottom = self.padding[2];
    let p_w_right = self.padding[3];
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
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

            for x in 0..k_h {
              for y in 0..k_w {
                let x_in = i * s_h + x - p_h_top;
                let y_in = j * s_w + y - p_w_left;
                let input_idx = x * k_w + y;

                let input_idx = input_idx as usize;

                if x_in >= 0 && x_in < H && y_in >= 0 && y_in < W {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  let input = &inputs[input_idx][[0, x_in * (W as usize) + y_in]];
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
        raw: vec![Fr::zero(); C_o],
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
    let H = self.input_shape[0];
    let W = self.input_shape[1];
    let k_h = self.kernel_shape[0];
    let k_w = self.kernel_shape[1];
    let s_h = self.stride[0];
    let s_w = self.stride[1];
    let p_h_top = self.padding[0];
    let p_w_left = self.padding[1];
    let p_h_bottom = self.padding[2];
    let p_w_right = self.padding[3];
    let H_o = (H - k_h + p_h_top + p_h_bottom) / s_h + 1;
    let W_o = (W - k_w + p_w_left + p_w_right) / s_w + 1;
    (0..H_o)
      .into_par_iter()
      .flat_map(|i| {
        (0..W_o)
          .map(move |j| {
            let mut sum = G1Projective::zero();
            for x in 0..k_h {
              for y in 0..k_w {
                let x_in = i * s_h + x - p_h_top;
                let y_in = j * s_w + y - p_w_left;
                let input_idx = (x * k_w + y) as usize;
                if x_in >= 0 && x_in < H && y_in >= 0 && y_in < W {
                  let x_in = x_in as usize;
                  let y_in = y_in as usize;
                  sum += &inputs[input_idx][[0, x_in * (W as usize) + y_in]].g1;
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
