#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use crate::MatMulFixedBasicBlock;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::BigInt;
use ark_ff::BigInteger;
use ark_ff::Field;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::{One, UniformRand, Zero};
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

pub struct BitDecompositionBasicBlock {
  pub mm_bb: Box<MatMulFixedBasicBlock>,
}
impl BasicBlock for BitDecompositionBasicBlock {
  // input: vec vec of fr
  // output: bit decomp
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let r = inputs
      .iter()
      .flat_map(|x| {
        x.iter()
          .map(|i| {
            let n: BigInt<4> = (*i).into();
            let bits = n.to_bits_le();
            bits.iter().map(|x| if *x { Fr::one() } else { Fr::zero() }).collect()
          })
          .collect::<Vec<_>>()
      })
      .collect();

    r
  }
  fn setup(&self, srs: &SRS, model: &Vec<&Data>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    self.mm_bb = Box::new(MatMulFixedBasicBlock);
    self.mm_bb.setup(srs, model)
  }
  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    // 0. output commit m*n msm 256
    // 1. input single commit m*n msm 1
    // 2. cqlin 8 msms 256

    // 3. b(1-b) = 0 ifft m*n 256,
    // 4. group all of them m*n msm 256
    // 5. rlc group again msm m*n 4 msm m

    // alternatively do aurora matmul b(1-b) ...
    // need to construct commitments to 1-b, m*n msm 256
    // input x b, (1-b) x 1
    // 4 msm 256, 3 msm 1
    // eval each on random beta, should equal 0?

    // does verifier need to know how to compute b(1-b) from b commitment
    // bn254 has 254 bits, so model is cols of 2^0, 2^1, ..., 2^253, 0, 0, 0
    // output is bit vec vec (decomps in each row). the data will be a
    // reshape inputs to be a column
    let mut input_col = vec![];
    for i in 0..inputs.len() {
      for j in 0..inputs[i].raw.len() {
        input_col.push(&Data::new(srs, &vec![inputs[i].raw[j]]));
      }
    }
    let sum_proof = self.mm_bb.prove(srs, setup, model, outputs, &input_col, rng);

    // b(1-b) + t b(1-b) + t^2 b(1-b) + ... = 0 256*m*n
    // r * b(1-b) + rt b(1-b) + rt^2(1-b) + ... = 0
    let alpha = Fr::rand(rng);

    let domain = GeneralEvaluationDomain::<Fr>::new(256).unwrap();
    let temp: Vec<Vec<_>> = outputs.into_par_iter().map(|x| (**x).raw.iter().map(|b| *b * (Fr::one() - *b)).collect()).collect();
    let temp = temp.into_par_iter().map(|x| domain.ifft(&x));
    // compute products b(1-b)
  }
  fn verify(
    &self,
    srs: &SRS,
    model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
  ) {
    // b_i(1 - b_i) = 0

    // sum of bits * power is
  }
}
