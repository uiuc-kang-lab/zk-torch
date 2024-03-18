#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use rand::{rngs::StdRng, SeedableRng};

// Takes in A,B and intertwines them into C
pub struct AlternateBasicBlock;
impl BasicBlock for AlternateBasicBlock {
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let n = inputs[0].len();
    let mut C = vec![];
    for i in 0..n {
      C.push(inputs[0][i]);
      C.push(inputs[1][i]);
    }
    vec![C]
  }
}

// Takes in C and splits it into evens A and odds B
pub struct SplitBasicBlock;
impl BasicBlock for SplitBasicBlock {
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let n2 = inputs[0].len();
    let mut A = vec![];
    let mut B = vec![];
    for i in 0..n2 {
      if i % 2 == 0 {
        A.push(inputs[0][i]);
      } else {
        B.push(inputs[0][i]);
      }
    }
    vec![A, B]
  }
}
