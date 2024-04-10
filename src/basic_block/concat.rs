#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util::{self, calc_pow};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use rand::{rngs::StdRng, SeedableRng};

pub struct ConcatBasicBlock;
// inputs are rows to A, output is concatenated rows of A
impl BasicBlock for ConcatBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![2])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let n = inputs.len();
    let m = inputs[0].len();
    let mut r = vec![Fr::zero(); n * m];
    for i in 0..n {
      for j in 0..m {
        r[i * m + j] = inputs[i][j];
      }
    }
    vec![r]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let n = inputs.len();
    let m = inputs[0].raw.len();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_nm = GeneralEvaluationDomain::<Fr>::new(n * m).unwrap();
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, n);
    let beta_pow = calc_pow(beta, m);
    let beta_pow = Data::new(srs, &beta_pow); //r is ignored
    let alpha_beta_pow: Vec<Fr> = (0..n * m)
      .map(|i| {
        let j = i / m;
        let k = i % m;
        alpha_pow[j] * beta_pow.raw[k]
      })
      .collect();
    let alpha_beta_pow = Data::new(srs, &alpha_beta_pow); //r is ignored

    let mut flat_A = vec![Fr::zero(); m];
    let mut flat_A_r = Fr::zero();
    for i in 0..n {
      for j in 0..m {
        flat_A[j] += inputs[i].raw[j] * alpha_pow[i];
      }
      flat_A_r += inputs[i].r * alpha_pow[i];
    }
    let mut flat_A = Data::new(srs, &flat_A);
    flat_A.r = flat_A_r;

    // Calculate Left
    let left_raw: Vec<Fr> = (0..m).map(|i| flat_A.raw[i] * beta_pow.raw[i]).collect();
    let left_poly = DensePolynomial {
      coeffs: domain_m.ifft(&left_raw),
    };
    let left_x = util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs);
    let left_Q_poly = flat_A.poly.mul(&beta_pow.poly).sub(&left_poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let left_Q_x = util::msm::<G1Projective>(&srs.X1A, &left_Q_poly.coeffs);
    let left_zero = srs.X1A[0] * (Fr::from(m as u32).inverse().unwrap() * left_raw.iter().sum::<Fr>());
    let left_zero_div = util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs[1..]);

    // Calculate Right
    let right_raw: Vec<Fr> = (0..n * m).map(|i| outputs[0].raw[i] * alpha_beta_pow.raw[i]).collect();
    let right_poly = DensePolynomial {
      coeffs: domain_nm.ifft(&right_raw),
    };
    let right_x = util::msm::<G1Projective>(&srs.X1A, &right_poly.coeffs);
    let right_Q_poly = outputs[0].poly.mul(&alpha_beta_pow.poly).sub(&right_poly).divide_by_vanishing_poly(domain_nm).unwrap().0;
    let right_Q_x = util::msm::<G1Projective>(&srs.X1A, &right_Q_poly.coeffs);
    let right_zero_div = util::msm::<G1Projective>(&srs.X1A, &right_poly.coeffs[1..]);

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..7).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let mut corr = vec![
      -(srs.X1P[m] - srs.X1P[0]) * r[1] + beta_pow.g1 * flat_A.r - srs.X1P[0] * r[0],
      -srs.X1P[1] * r[3] + srs.X1P[0] * (r[0] - r[2]),
      -(srs.X1P[n * m] - srs.X1P[0]) * r[5] + alpha_beta_pow.g1 * outputs[0].r - srs.X1P[0] * r[4],
      -srs.X1P[1] * r[6] + srs.X1P[0] * (r[4] - r[2] * Fr::from(n as u32).inverse().unwrap()),
    ];
    proof.append(&mut corr);

    return (proof, vec![]);
  }
  fn verify(
    &self,
    srs: &SRS,
    model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let n = inputs.len();
    let m = inputs[0].len;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_nm = GeneralEvaluationDomain::<Fr>::new(n * m).unwrap();
    let [left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div, corr1, corr2, corr3, corr4] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, n);
    let beta_pow = calc_pow(beta, m);
    let beta_pow_coeff = domain_m.ifft(&beta_pow);
    let beta_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.X2A, &beta_pow_coeff).into();
    let alpha_beta_pow: Vec<Fr> = (0..n * m)
      .map(|i| {
        let j = i / m;
        let k = i % m;
        alpha_pow[j] * beta_pow[k]
      })
      .collect();
    let alpha_beta_pow_coeff = domain_nm.ifft(&alpha_beta_pow);
    let alpha_beta_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.X2A, &alpha_beta_pow_coeff).into();

    // Calculate flat_A
    let temp: Vec<_> = (0..n).map(|i| inputs[i].g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Check left(x) (left_i = flat_A_i * beta_pow_i)
    let lhs = Bn254::pairing(flat_A_g1, beta_pow_g2) - Bn254::pairing(left_x, srs.X2A[0]);
    let rhs = Bn254::pairing(left_Q_x, srs.X2A[m] - srs.X2A[0]) + Bn254::pairing(corr1, srs.Y2A);
    assert!(lhs == rhs);

    // Check left(x) - left(0) is divisible by x
    let lhs = Bn254::pairing(left_x - left_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(left_zero_div, srs.X2A[1]) + Bn254::pairing(corr2, srs.Y2A);
    assert!(lhs == rhs);

    // Check right(x) (right_i = flat_C_i * beta_pow_i)
    let lhs = Bn254::pairing(outputs[0].g1, alpha_beta_pow_g2) - Bn254::pairing(right_x, srs.X2A[0]);
    let rhs = Bn254::pairing(right_Q_x, srs.X2A[n * m] - srs.X2A[0]) + Bn254::pairing(corr3, srs.Y2A);
    assert!(lhs == rhs);

    // Assume right(0) = left(0)*n (which assumes ∑left=∑right)
    let right_zero: G1Affine = (left_zero * (Fr::from(n as u32).inverse().unwrap())).into();

    //check right(x) - right(0) is divisible by x
    let lhs = Bn254::pairing(right_x - right_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(right_zero_div, srs.X2A[1]) + Bn254::pairing(corr4, srs.Y2A);
    assert!(lhs == rhs);
  }
}
