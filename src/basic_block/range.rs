#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::{
  onnx,
  util::{self, calc_pow},
};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::Field;
use ark_poly::{
  evaluations::univariate::Evaluations, univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
};
use ark_serialize::CanonicalSerialize;
use ark_std::{
  ops::{Add, Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::{arr1, indices, ArrayD, ArrayView, ArrayView1, ArrayViewD, Axis, Dim, Dimension, IxDyn, IxDynImpl, NdIndex, Shape, Zip};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::{
  cmp::{max, min},
  collections::HashMap,
  iter::{once, repeat},
};

// RangeConstBasicBlock is a basic block that creates a tensor of a range of values.
// The range is defined by three constants: the start, limit, and delta values.
#[derive(Debug)]
pub struct RangeConstBasicBlock {
  pub start: i32,
  pub limit: i32,
  pub delta: i32,
}
impl BasicBlock for RangeConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let element_num = max(0, ((self.limit - self.start) + self.delta - 1) / self.delta);
    let mut r = vec![];
    let mut x = self.start;
    while x < self.limit {
      r.push(Fr::from(x));
      x += self.delta;
    }
    let element_num_pad = util::next_pow(element_num as u32) as usize;
    while r.len() < element_num_pad {
      r.push(Fr::zero());
    }
    vec![arr1(&r).into_dyn()]
  }

  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let element_num = max(0, ((self.limit - self.start) + self.delta - 1) / self.delta);
    let element_num_pad = util::next_pow(element_num as u32) as usize;
    let domain = GeneralEvaluationDomain::<Fr>::new(element_num_pad.clone() as usize).unwrap();

    let mut r = vec![];
    let mut x = self.start;
    while x < self.limit {
      r.push(Fr::from(x));
      x += self.delta;
    }
    while r.len() < element_num_pad {
      r.push(Fr::zero());
    }
    let range_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&r));
    let range_x = util::msm::<G1Projective>(&srs.X1A, &range_poly.coeffs);
    (vec![range_x], vec![], vec![])
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let C = srs.Y1P * outputs[0].first().unwrap().r;
    (vec![setup.0[0].into(), C.into()], vec![], vec![])
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    assert!(proof.0[0] + proof.0[1] == outputs[0].first().unwrap().g1);
    vec![]
  }
}
