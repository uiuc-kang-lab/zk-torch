use crate::{
  basic_block::{Data, DataEnc, SRS},
  util::{self, convert_to_data},
  PairingCheck, ProveVerifyCache,
};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{cmp::max, UniformRand};
use ndarray::Axis;
use std::{
  borrow::Borrow,
  ops::{Mul, Sub},
  time::Instant,
};

use super::BasicBlock;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_std::Zero;
use ndarray::{arr0, arr1, azip, Array, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct MaxV1BasicBlock;

// This max includes a proof and is intended to be followed by a lookup range check.
impl BasicBlock for MaxV1BasicBlock {
  // Returns the max of the input and max - x for all x in input
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    let max_arr = inputs[0]
      .fold_axis(Axis(0), Fr::zero(), |max, y| if *y < Fr::from(1 << 28) && *y > *max { *y } else { *max })
      .into_shape(vec![1])
      .unwrap();
    let max_val = max_arr.first().unwrap();

    let mut r = ArrayD::zeros(inputs[0].shape());
    azip!((r in &mut r, &x in inputs[0]) *r = *max_val - x);
    vec![max_arr, r]
  }

  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let N = outputs[1].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Diff proving
    let mut proof_1 = vec![];
    let max = outputs[0].first().unwrap();
    let input = inputs[0].first().unwrap();
    let diff = outputs[1].first().unwrap();
    let C = srs.X1P[0] * (max.r - input.r - diff.r);
    proof_1.push(C);

    // Multiplication proving
    let mut rng = StdRng::from_entropy();

    let mut product = outputs[1].first().unwrap().clone();
    let mut proof_2 = vec![];
    for i in 1..outputs[1].len() {
      let diff = &outputs[1][i];

      if i < outputs[1].len() - 1 {
        let new_product_raw: Vec<Fr> = diff.raw.iter().zip(product.raw.iter()).map(|(x, y)| x * y).collect();
        let new_product = Data::new(srs, &new_product_raw);

        let t = diff.poly.mul(&product.poly).sub(&new_product.poly).divide_by_vanishing_poly(domain).unwrap().0;

        let r = Fr::rand(&mut rng);

        let tx = util::msm::<G2Projective>(&srs.X2A, &t.coeffs) + srs.Y2P * r;

        let C = (product.g1 * diff.r) + (diff.g1 * product.r) + srs.Y1P * (product.r * diff.r)
          - (srs.X1P[0] * new_product.r)
          - ((srs.X1P[N] - srs.X1P[0]) * r);
        proof_2.push(tx);
        proof_1.push(new_product.g1 + srs.Y1P * new_product.r);
        proof_1.push(C);

        product = new_product.clone();
      } else {
        let t = diff.poly.mul(&product.poly).divide_by_vanishing_poly(domain).unwrap().0;

        let r = Fr::rand(&mut rng);
        let tx = util::msm::<G2Projective>(&srs.X2A, &t.coeffs) + srs.Y2P * r;

        let C = (product.g1 * diff.r) + (diff.g1 * product.r) + srs.Y1P * (product.r * diff.r) - ((srs.X1P[N] - srs.X1P[0]) * r);
        proof_2.push(tx);
        proof_1.push(C);
      }
    }

    (proof_1, proof_2, Vec::new())
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let n = outputs[1].len();
    let max = outputs[0].first().unwrap();
    let input = inputs[0].first().unwrap();
    let diff = outputs[1].first().unwrap();

    // Verify that diff = max - input
    checks.push(vec![((max.g1 - input.g1 - diff.g1).into(), srs.X2A[0]), (-proof.0[0], srs.Y2A)]);

    let mut product = diff.g1;
    for i in 0..n - 1 {
      let diff = &outputs[1][i + 1];
      if i < n - 2 {
        let tx = proof.0[n + 3 * i];
        let new_product = proof.0[n + 3 * i + 1];
        let C = proof.0[n + 3 * i + 2];
        let gx2 = proof.1[i];

        // Verify f(x)*g(x)-h(x)=z(x)t(x)
        checks.push(vec![
          (product, gx2),
          (-new_product, srs.X2A[0]),
          (-tx, (srs.X2A[diff.len] - srs.X2A[0]).into()),
          (-C, srs.Y2A),
        ]);

        // Verify gx2
        checks.push(vec![(diff.g1, srs.X2A[0]), (-srs.X1A[0], gx2)]);
        product = new_product;
      } else {
        let tx = proof.0[n + 3 * i];
        let C = proof.0[n + 3 * i + 1];
        let gx2 = proof.1[i];

        // Verify f(x)*g(x)-h(x)=z(x)t(x)
        checks.push(vec![(product, gx2), (-tx, (srs.X2A[diff.len] - srs.X2A[0]).into()), (-C, srs.Y2A)]);

        // Verify gx2
        checks.push(vec![(diff.g1, srs.X2A[0]), (-srs.X1A[0], gx2)]);
      }
    }
    checks
  }
}
