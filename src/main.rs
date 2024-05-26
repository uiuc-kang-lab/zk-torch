use ark_ff::BigInt;
use ark_ff::BigInteger;
use ark_ff::One;
use ark_ff::Zero;
use ark_poly::univariate::DensePolynomial;
use layer::conv::splat_input;
use ndarray::ArrayD;
use ndarray::Dim;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{cmp::min, collections::HashMap, time::Instant};

use ark_bn254::{
  g1::{G1_GENERATOR_X, G1_GENERATOR_Y},
  g2::{G2_GENERATOR_X, G2_GENERATOR_Y},
  Fr, G1Affine, G1Projective, G2Affine, G2Projective,
};
use ark_ec::AffineRepr;
use ark_std::UniformRand;
use basic_block::*;
use ndarray::{Array, IxDyn};
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

use crate::basic_block::*;
use crate::layer::conv::out_hw;
use crate::layer::conv::splat_weights;
use crate::util::*;

mod basic_block;
mod graph;
mod layer;
mod onnx;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

// fn generate_srs(N: usize) -> (Vec<G1Affine>, Vec<G2Affine>) {
//   let mut srs = (vec![G1Affine::generator(); N], vec![G2Affine::generator(); N + 1]);
//   let mut rng = StdRng::from_entropy();
//   let x = Fr::rand(&mut rng);
//   let mut xp = x;
//   for i in 1..N {
//     srs.0[i] = (srs.0[i] * xp).into();
//     xp *= x;
//   }
//   xp = x;
//   for i in 1..N + 1 {
//     srs.1[i] = (srs.1[i] * xp).into();
//     xp *= x;
//   }
//   (srs.0, srs.1)
// }

fn testBasicBlock<BB: BasicBlock>(mut basic_block: BB, srs: &SRS, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let outputs = basic_block.run(model, inputs);
  let outputs: Vec<&ArrayD<Fr>> = outputs.iter().map(|x| x).collect();
  let model = convert_to_data(srs, model);
  let setup = basic_block.setup(srs, &model);
  let setup: (Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>) = (
    setup.0.iter().map(|y| (*y).into()).collect(),
    setup.1.iter().map(|y| (*y).into()).collect(),
    setup.2.iter().map(|y| (y.clone())).collect(),
  );
  let setup = (&setup.0, &setup.1, &setup.2);
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<Data>> = basic_block.encodeOutputs(srs, &model, &inputs, &outputs);
  let outputs: Vec<&ArrayD<Data>> = outputs.iter().map(|output| output).collect();
  let mut rng2 = rng.clone();
  let mut cache = HashMap::new();
  let proof = basic_block.prove(srs, setup, &model, &inputs, &outputs, &mut rng, &mut cache);
  // let mut rng2 = rng.clone();
  let start = Instant::now();
  let proof = basic_block.prove(srs, setup, &model, &inputs, &outputs, &mut rng, &mut cache);
  let prove_duration = start.elapsed();
  println!("prove duration: {:?}", prove_duration);
  // let proof: (Vec<G1Affine>, Vec<G2Affine>) = (
  //   proof.0.iter().map(|y| (*y).into()).collect(),
  //   proof.1.iter().map(|y| (*y).into()).collect(),
  // );
  // let proof = (&proof.0, &proof.1);
  // let model: Vec<_> = model.iter().map(|x| DataEnc::new(srs, x)).collect();
  // let model = model.iter().map(|x| x).collect();
  // let inputs: Vec<_> = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  // let inputs = inputs.iter().map(|x| x).collect();
  // let outputs: Vec<_> = outputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  // let outputs = outputs.iter().map(|x| x).collect();
  // basic_block.verify(srs, &model, &inputs, &outputs, proof, &mut rng2);
}

fn cq_bench(srs: &SRS) {
  let t = CQ_DIM;
  let a: Vec<_> = (0..t).into_par_iter().map(|i| Fr::from(i as u32)).collect();
  let a = ArrayD::from_shape_vec(IxDyn(&[t]), a).unwrap();
  let input: Vec<_> = (0..t)
    .into_par_iter()
    .map_init(rand::thread_rng, |rng, _| {
      let r = rng.gen_range(0..t);
      Fr::from(r as u32)
    })
    .collect();
  let input = ArrayD::from_shape_vec(IxDyn(&[t]), input).unwrap();
  testBasicBlock(CQBasicBlock { setup: None }, srs, &a, &vec![&input]);
}

fn conv_bench(srs: &SRS) {
  let mut rng = StdRng::from_entropy();
  let empty = ArrayD::zeros(IxDyn(&[0]));
  for i in 0..CONV_INPUTS.len() {
    let weights_dim = WEIGHTS_DIMS[i];
    let ci = weights_dim[1];
    let ch = weights_dim[2];
    let cw = weights_dim[3];
    let inputs = ArrayD::from_shape_fn(Dim(IxDyn(&CONV_INPUTS[i])), |_| Fr::from(rng.gen_range(-4..4)));
    let permutation = splat_input(&CONV_INPUTS[i].to_vec(), &STRIDES[i].to_vec(), &PADS[i].to_vec(), ci, ch, cw);
    let output_dim = permutation.dim();
    let output_dim2 = permutation.dim();
    println!("copy constraint: {:?} -> {:?}", inputs.dim(), output_dim);
    testBasicBlock(
      CopyConstraintBasicBlock {
        permutation,
        input_dim: inputs.dim(),
        output_dim,
      },
      srs,
      &empty,
      &vec![&inputs],
    );

    let splat_inputs = ArrayD::from_shape_fn(output_dim2, |_| Fr::from(rng.gen_range(-4..4)));
    let weights = ArrayD::from_shape_fn(Dim(IxDyn(&WEIGHTS_DIMS[i])), |_| Fr::from(rng.gen_range(-4..4)));
    let splat_weights = splat_weights(&weights);
    println!("cqlin: {:?} x {:?}", splat_inputs.dim(), splat_weights.dim());
    testBasicBlock(CQLinBasicBlock {}, srs, &splat_weights, &vec![&splat_inputs]);

    let bias = ArrayD::from_shape_fn(Dim(IxDyn(&[splat_weights.shape()[1]])), |_| Fr::from(rng.gen_range(-4..4)));
    let inputs = ArrayD::from_shape_fn(Dim(IxDyn(&[splat_inputs.shape()[0], splat_weights.shape()[1]])), |_| {
      Fr::from(rng.gen_range(-4..4))
    });
    println!("add: {:?} x {:?}", inputs.dim(), bias.dim());
    testBasicBlock(
      RepeaterBasicBlock {
        basic_block: Box::new(AddBasicBlock {}),
        N: 1,
      },
      srs,
      &empty,
      &vec![&inputs, &bias],
    );

    let padding = vec![[0, 0], [0, 0], [PADS[i][0], PADS[i][2]], [PADS[i][1], PADS[i][3]]];
    let (oh, ow) = out_hw(
      CONV_INPUTS[i][2],
      CONV_INPUTS[i][3],
      STRIDES[i][0],
      STRIDES[i][1],
      WEIGHTS_DIMS[i][2],
      WEIGHTS_DIMS[i][3],
      &padding,
    );
    let permutation = ArrayD::from_shape_fn(IxDyn(&[1, bias.shape()[0], oh, ow]), |_| IxDyn(&[0, 0]));
    let (m, n) = (oh.next_power_of_two(), ow.next_power_of_two());
    let padding = vec![[0, 0], [0, 0], [0, m - permutation.shape()[2]], [0, n - permutation.shape()[3]]];
    let permutation = layer::conv::pad(&permutation, &padding, &IxDyn(&[0, 0]));
    let output_dim = permutation.dim();
    println!("copy constraint: {:?} -> {:?}", inputs.dim(), output_dim);
    testBasicBlock(
      CopyConstraintBasicBlock {
        permutation,
        input_dim: inputs.dim(),
        output_dim,
      },
      srs,
      &empty,
      &vec![&inputs],
    );
  }
}

const CONV_INPUTS: [[usize; 4]; 52] = [
  [1, 4, 256, 256],
  [1, 32, 128, 128],
  [1, 32, 128, 128],
  [1, 16, 128, 128],
  [1, 128, 128, 128],
  [1, 128, 64, 64],
  [1, 32, 64, 64],
  [1, 256, 64, 64],
  [1, 256, 64, 64],
  [1, 32, 64, 64],
  [1, 256, 64, 64],
  [1, 256, 32, 32],
  [1, 32, 32, 32],
  [1, 256, 32, 32],
  [1, 256, 32, 32],
  [1, 32, 32, 32],
  [1, 256, 32, 32],
  [1, 256, 32, 32],
  [1, 32, 32, 32],
  [1, 256, 32, 32],
  [1, 256, 16, 16],
  [1, 64, 16, 16],
  [1, 256, 32, 32],
  [1, 512, 32, 32],
  [1, 64, 16, 16],
  [1, 512, 16, 16],
  [1, 512, 16, 16],
  [1, 64, 16, 16],
  [1, 512, 16, 16],
  [1, 512, 16, 16],
  [1, 64, 16, 16],
  [1, 512, 16, 16],
  [1, 512, 16, 16],
  [1, 128, 16, 16],
  [1, 1024, 16, 16],
  [1, 1024, 16, 16],
  [1, 128, 16, 16],
  [1, 1024, 16, 16],
  [1, 1024, 16, 16],
  [1, 128, 16, 16],
  [1, 1024, 16, 16],
  [1, 1024, 8, 8],
  [1, 256, 8, 8],
  [1, 1024, 8, 8],
  [1, 1024, 8, 8],
  [1, 256, 8, 8],
  [1, 1024, 8, 8],
  [1, 1024, 8, 8],
  [1, 256, 8, 8],
  [1, 1024, 8, 8],
  [1, 1024, 8, 8],
  [1, 512, 8, 8],
  // [1, 3, 8, 8],
];

const WEIGHTS_DIMS: [[usize; 4]; 52] = [
  [32, 3, 3, 3],
  [32, 1, 3, 3],
  [16, 32, 1, 1],
  [96, 16, 1, 1],
  [96, 1, 3, 3],
  [24, 96, 1, 1],
  [144, 24, 1, 1],
  [144, 1, 3, 3],
  [24, 144, 1, 1],
  [144, 24, 1, 1],
  [144, 1, 3, 3],
  [32, 144, 1, 1],
  [192, 32, 1, 1],
  [192, 1, 3, 3],
  [32, 192, 1, 1],
  [192, 32, 1, 1],
  [192, 1, 3, 3],
  [32, 192, 1, 1],
  [192, 32, 1, 1],
  [192, 1, 3, 3],
  [64, 192, 1, 1],
  [384, 64, 1, 1],
  [384, 1, 3, 3],
  [64, 384, 1, 1],
  [384, 64, 1, 1],
  [384, 1, 3, 3],
  [64, 384, 1, 1],
  [384, 64, 1, 1],
  [384, 1, 3, 3],
  [64, 384, 1, 1],
  [384, 64, 1, 1],
  [384, 1, 3, 3],
  [96, 384, 1, 1],
  [576, 96, 1, 1],
  [576, 1, 3, 3],
  [96, 576, 1, 1],
  [576, 96, 1, 1],
  [576, 1, 3, 3],
  [96, 576, 1, 1],
  [576, 96, 1, 1],
  [576, 1, 3, 3],
  [160, 576, 1, 1],
  [960, 160, 1, 1],
  [960, 1, 3, 3],
  [160, 960, 1, 1],
  [960, 160, 1, 1],
  [960, 1, 3, 3],
  [160, 960, 1, 1],
  [960, 160, 1, 1],
  [960, 1, 3, 3],
  [320, 960, 1, 1],
  [1280, 320, 1, 1],
];

const PADS: [[usize; 4]; 52] = [
  [0, 0, 1, 1],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];

const STRIDES: [[usize; 2]; 52] = [
  [2, 2],
  [1, 1],
  [1, 1],
  [1, 1],
  [2, 2],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [2, 2],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [2, 2],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [2, 2],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
  [1, 1],
];

const MUL_DIMS: [(usize, usize, usize, usize); 8] = [
  (16384, 64, 64, 1),
  (4096, 1024, 64, 6),
  (1024, 1024, 128, 2),
  (1024, 2048, 128, 6),
  (64, 2048, 256, 2),
  (256, 4096, 256, 10),
  (64, 4096, 512, 2),
  (64, 8192, 512, 4),
  // (1, 8, 8, 1),
  // (1, 16, 16, 1),
  // (1, 32, 32, 1),
  // (1, 64, 64, 1),
  // (1, 128, 128, 1),
  // (1, 256, 256, 1),
  // (1, 512, 512, 1),
  // (1, 1024, 1024, 1),
];

const CQ_DIM: usize = 65536;

// const PLOOKUP_DIMS: [(usize, usize); 4] = [
//   (1048575, 1), // 128 x 128 x 64
//   (262143, 5),  // 64 x 64 x 64
//   (131071, 4),
//   (65535, 4),
//   // (32768, 3),
// ];

// fn matmul_bench(srs: &SRS) {
//   for n in MUL_DIMS {
//     println!("{:?}", n);
//     let l = n.0;
//     let m = n.1;
//     let n = n.2;
//     let mut matrix: Vec<Vec<Fr>> = vec![];
//     for _ in 0..m {
//       matrix.push((0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect());
//     }
//     let matrix: Vec<_> = matrix.iter().map(|x| x).collect();
//     let mut inputs: Vec<Vec<Fr>> = vec![];
//     for _ in 0..l {
//       inputs.push((0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect());
//     }
//     let inputs: Vec<_> = inputs.iter().map(|x| x).collect();
//     testBasicBlock(CQLinBasicBlock {}, &srs, &matrix, &inputs);
//   }
// }

fn main() {
  let srs = ptau::load_file("powersOfTau28_hez_final_20.ptau", 20, 20);

  // matmul_bench(&srs);
  conv_bench(&srs);
  // cq_bench(&srs);
}
