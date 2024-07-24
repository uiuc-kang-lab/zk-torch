#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::{
  onnx,
  util::{self, calc_pow},
  PairingCheck, ProveVerifyCache,
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
    let mut r = vec![];
    let mut x = self.start;
    while x < self.limit {
      r.push(Fr::from(x));
      x += self.delta;
    }
    vec![arr1(&r).into_dyn()]
  }
}

// RangeBasicBlock is a basic block that creates a tensor of a range of values.
// The difference between RangeBasicBlock and RangeConstBasicBlock is that RangeBasicBlock
// takes the limit value as a private input, while RangeConstBasicBlock takes the limit value as a constant.
// TODO: add proper blinding for opening arguments, similar issue as in CopyConstraintBasicBlock
#[derive(Debug)]
pub struct RangeBasicBlock {
  pub start: i32,
  pub delta: i32,
}
impl BasicBlock for RangeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    let limit = util::fr_to_int(inputs[0][0]) as i32;
    let mut r = vec![];
    let mut x = self.start;
    while x < limit {
      r.push(Fr::from(x));
      x += self.delta;
    }
    vec![arr1(&r).into_dyn()]
  }

  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let limit = util::fr_to_int(inputs[0].first().unwrap().raw[0]) as i32;
    // number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
    let element_num = max(0, ((limit - self.start) + self.delta - 1) / self.delta);
    let N = onnx::CQ_RANGE;
    assert!(element_num <= N as i32);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let step_poly = DensePolynomial::from_coefficients_vec(vec![Fr::from(self.delta)]);
    let mut selection = vec![Fr::zero(); N];
    for i in 0..(element_num - 1) {
      selection[i as usize] = Fr::one();
    }
    let selection_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&selection));

    let f_poly = outputs[0].first().unwrap().poly.clone();
    let f_coeffs = f_poly.coeffs.clone();
    let mut omega_gen = Fr::from(1);
    let f_omega_coeffs = f_coeffs
      .iter()
      .map(|x| {
        let mut y = x.clone();
        y = y * omega_gen;
        omega_gen = omega_gen * omega;
        y
      })
      .collect();
    let f_omega_poly = DensePolynomial { coeffs: f_omega_coeffs };
    let t_poly: DensePolynomial<Fr> = f_poly.add(step_poly).sub(&f_omega_poly).mul(&selection_poly).divide_by_vanishing_poly(domain).unwrap().0;

    let step_poly_x = util::msm::<G1Projective>(&srs.X1A, &[Fr::from(self.delta)]);
    let selection_poly_x = util::msm::<G1Projective>(&srs.X1A, &selection_poly.coeffs);
    let t_poly_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);
    let mut proof0 = vec![step_poly_x, selection_poly_x, t_poly_x];

    let mut bytes = Vec::new();
    proof0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let z = Fr::rand(rng);

    let f_omega_z = outputs[0].first().unwrap().poly.clone().evaluate(&(omega * z));
    let f_z = outputs[0].first().unwrap().poly.evaluate(&z);
    let selection_z = selection_poly.evaluate(&z);
    let step_z = Fr::from(self.delta);
    let t_z = t_poly.evaluate(&z);
    let proof2 = vec![f_omega_z, f_z, step_z, selection_z, t_z];

    let mut bytes = Vec::new();
    proof2.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let gamma = Fr::rand(rng);
    let beta = Fr::rand(rng);

    // calculate h = [(f(x) - f(z)) + gamma * (step(x) - step(z)) + gamma^2 * (selection(x) - selection(z)) + gamma^3 * (t(x) - t(z))] / (x - z)
    // h_denominator = x - z
    let h_denominator = DensePolynomial::from_coefficients_vec(vec![-z, Fr::one()]);
    let f_z_poly = DensePolynomial { coeffs: vec![f_z] };
    let selection_z_poly = DensePolynomial { coeffs: vec![selection_z] };
    let t_z_poly = DensePolynomial { coeffs: vec![t_z] };

    let gamma_pow = calc_pow(gamma, 3);
    let f_minus_f_z_poly = outputs[0].first().unwrap().poly.sub(&f_z_poly);
    let step_minus_step_z_poly = DensePolynomial::from_coefficients_vec(vec![Fr::zero()]);
    // let step_minus_step_z_poly = DensePolynomial::from_coefficients_vec(step_minus_step_z_poly.coeffs.iter().map(|x| x * &gamma_pow[0]).collect());
    let selection_minus_selection_z_poly = selection_poly.sub(&selection_z_poly);
    let selection_minus_selection_z_poly =
      DensePolynomial::from_coefficients_vec(selection_minus_selection_z_poly.coeffs.iter().map(|x| x * &gamma_pow[1]).collect());
    let t_minus_t_z_poly = t_poly.sub(&t_z_poly);
    let t_minus_t_z_poly = DensePolynomial::from_coefficients_vec(t_minus_t_z_poly.coeffs.iter().map(|x| x * &gamma_pow[2]).collect());
    let h_numerator = f_minus_f_z_poly.add(step_minus_step_z_poly).add(selection_minus_selection_z_poly).add(t_minus_t_z_poly);

    let h_poly = &h_numerator / &h_denominator;
    let h_poly_x = util::msm::<G1Projective>(&srs.X1A, &h_poly.coeffs);
    proof0.push(h_poly_x);

    // calculate h' = [f(x) - f(omega * z)] / (x - omega * z)
    // h_prime_denominator =  x - omega * z
    let h_prime_denominator = DensePolynomial::from_coefficients_vec(vec![-z * omega, Fr::one()]);
    let f_omega_z_poly = DensePolynomial { coeffs: vec![f_omega_z] };
    let h_prime_numerator = outputs[0].first().unwrap().poly.sub(&f_omega_z_poly);

    let h_prime_poly = &h_prime_numerator / &h_prime_denominator;
    let h_prime_poly_x = util::msm::<G1Projective>(&srs.X1A, &h_prime_poly.coeffs);
    proof0.push(h_prime_poly_x);

    // blinding
    let C = srs.X1P[0] * (outputs[0].first().unwrap().r * (beta + Fr::one()));
    proof0.push(C);

    (proof0, vec![], proof2)
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let N = onnx::CQ_RANGE;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // First check selection(z) * [f(z) + step(z) - f(omega * z)] == t(z) * vanishing_poly(z)
    let proof0_for_check = &proof.0[..3];
    let mut bytes = Vec::new();
    proof0_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let z = Fr::rand(rng);

    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_poly_z = vanishing_poly.evaluate(&z);
    let [f_omega_z, f_z, step_z, selection_z, t_z] = proof.2[..5] else {
      panic!("Invalid proof length");
    };

    assert!(selection_z * (f_z + step_z - f_omega_z) == t_z * vanishing_poly_z);

    // Then check openings are correct

    let proof2_for_check = &proof.2[..5];
    let mut bytes = Vec::new();
    proof2_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let gamma = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let omega = domain.group_gen();
    let [step_poly_x, selection_poly_x, t_poly_x, h_poly_x, h_prime_poly_x, C] = proof.0[..6] else {
      panic!("Invalid proof length");
    };

    let gamma_pow = calc_pow(gamma, 3);
    let mut check_for_opening_at_z =
      outputs[0].first().unwrap().g1 + step_poly_x * gamma_pow[0] + selection_poly_x * gamma_pow[1] + t_poly_x * gamma_pow[2];
    check_for_opening_at_z -= srs.X1A[0] * (f_z + step_z * gamma_pow[0] + selection_z * gamma_pow[1] + t_z * gamma_pow[2]);
    let mut check_for_opening_at_omega_z: G1Projective = outputs[0].first().unwrap().g1.into();
    check_for_opening_at_omega_z -= srs.X1A[0] * f_omega_z;
    let mut checks = Vec::new();
    checks.push(vec![
      ((h_poly_x + h_prime_poly_x * beta).into(), srs.X2A[1]),
      (
        (-(check_for_opening_at_z + check_for_opening_at_omega_z * beta + h_poly_x * z + h_prime_poly_x * (beta * omega * z))).into(),
        srs.X2A[0],
      ),
      (C.into(), srs.Y2A),
    ]);
    checks
  }
}
