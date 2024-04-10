#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use rand::{rngs::StdRng, SeedableRng};

// Start at alpha^1
fn calc_pow(alpha: Fr, n: usize) -> Vec<Fr> {
  let mut pow: Vec<Fr> = vec![alpha; n];
  for i in 0..n - 1 {
    pow[i + 1] = pow[i] * alpha;
  }
  pow
}

pub struct AlternatingBasicBlock;
// Inputs are X,Y,Z where X alternating with Y equals Z
// X,Y,Z are multiplied by alpha,beta,alpha_beta to get A,B,C
impl BasicBlock for AlternatingBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1, 1, 1])
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    _outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let n = inputs[0].raw.len();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2n = GeneralEvaluationDomain::<Fr>::new(n * 2).unwrap();
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, n);
    let alpha_pow = Data::new(srs, &alpha_pow);
    let beta_pow = calc_pow(beta, n);
    let beta_pow = Data::new(srs, &beta_pow);
    let alpha_beta_pow: Vec<Fr> = (0..2 * n).map(|i| if i % 2 == 0 { alpha_pow.raw[i / 2] } else { beta_pow.raw[i / 2] }).collect();
    let alpha_beta_pow = Data::new(srs, &alpha_beta_pow);

    // Calculate A
    let A_raw: Vec<Fr> = (0..n).map(|i| inputs[0].raw[i] * alpha_pow.raw[i]).collect();
    let A_poly = DensePolynomial {
      coeffs: domain_n.ifft(&A_raw),
    };
    let A_x = util::msm::<G1Projective>(&srs.X1A, &A_poly.coeffs);
    let A_Q_poly = inputs[0].poly.mul(&alpha_pow.poly).sub(&A_poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let A_Q_x = util::msm::<G1Projective>(&srs.X1A, &A_Q_poly.coeffs);
    let A_zero = srs.X1A[0] * (Fr::from(n as u32).inverse().unwrap() * A_raw.iter().sum::<Fr>());
    let A_zero_div = util::msm::<G1Projective>(&srs.X1A, &A_poly.coeffs[1..]);

    // Calculate B
    let B_raw: Vec<Fr> = (0..n).map(|i| inputs[1].raw[i] * beta_pow.raw[i]).collect();
    let B_poly = DensePolynomial {
      coeffs: domain_n.ifft(&B_raw),
    };
    let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
    let B_Q_poly = inputs[1].poly.mul(&beta_pow.poly).sub(&B_poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
    let B_zero = srs.X1A[0] * (Fr::from(n as u32).inverse().unwrap() * B_raw.iter().sum::<Fr>());
    let B_zero_div = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..]);

    // Calculate C
    let C_raw: Vec<Fr> = (0..n * 2).map(|i| inputs[2].raw[i] * alpha_beta_pow.raw[i]).collect();
    let C_poly = DensePolynomial {
      coeffs: domain_2n.ifft(&C_raw),
    };
    let C_x = util::msm::<G1Projective>(&srs.X1A, &C_poly.coeffs);
    let C_Q_poly = inputs[2].poly.mul(&alpha_beta_pow.poly).sub(&C_poly).divide_by_vanishing_poly(domain_2n).unwrap().0;
    let C_Q_x = util::msm::<G1Projective>(&srs.X1A, &C_Q_poly.coeffs);
    let C_zero_div = util::msm::<G1Projective>(&srs.X1A, &C_poly.coeffs[1..]);

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..11).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero, B_zero_div, C_x, C_Q_x, C_zero_div];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let mut corr = vec![
      -(srs.X1P[n] - srs.X1P[0]) * r[1] + alpha_pow.g1 * inputs[0].r - srs.X1P[0] * r[0],
      -srs.X1P[1] * r[3] + srs.X1P[0] * (r[0] - r[2]),
      -(srs.X1P[n] - srs.X1P[0]) * r[5] + beta_pow.g1 * inputs[1].r - srs.X1P[0] * r[4],
      -srs.X1P[1] * r[7] + srs.X1P[0] * (r[4] - r[6]),
      -(srs.X1P[n * 2] - srs.X1P[0]) * r[9] + alpha_beta_pow.g1 * inputs[2].r - srs.X1P[0] * r[8],
      -srs.X1P[1] * r[10] + srs.X1P[0] * (r[8] - (r[2] + r[6]) * Fr::from(2).inverse().unwrap()),
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
    let n = inputs[0].len;
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2n = GeneralEvaluationDomain::<Fr>::new(n * 2).unwrap();
    let [A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero, B_zero_div, C_x, C_Q_x, C_zero_div, corr1, corr2, corr3, corr4, corr5, corr6] =
      proof.0[..]
    else {
      panic!("Wrong proof format")
    };

    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, n); //r is ignored
    let alpha_pow_coeff = domain_n.ifft(&alpha_pow);
    let alpha_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.X2A, &alpha_pow_coeff).into();
    let beta_pow = calc_pow(beta, n); //r is ignored
    let beta_pow_coeff = domain_n.ifft(&beta_pow);
    let beta_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.X2A, &beta_pow_coeff).into();
    let alpha_beta_pow: Vec<Fr> = (0..2 * n).map(|i| if i % 2 == 0 { alpha_pow[i / 2] } else { beta_pow[i / 2] }).collect();
    let alpha_beta_pow_coeff = domain_2n.ifft(&alpha_beta_pow);
    let alpha_beta_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.X2A, &alpha_beta_pow_coeff).into();

    // Sumcheck A
    let lhs = Bn254::pairing(inputs[0].g1, alpha_pow_g2) - Bn254::pairing(A_x, srs.X2A[0]);
    let rhs = Bn254::pairing(A_Q_x, srs.X2A[n] - srs.X2A[0]) + Bn254::pairing(corr1, srs.Y2A);
    assert!(lhs == rhs);

    // Check A(x) - A(0) is divisible by x
    let lhs = Bn254::pairing(A_x - A_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(A_zero_div, srs.X2A[1]) + Bn254::pairing(corr2, srs.Y2A);
    assert!(lhs == rhs);

    // CHECKPOINT

    // Sumcheck B
    let lhs = Bn254::pairing(inputs[1].g1, beta_pow_g2) - Bn254::pairing(B_x, srs.X2A[0]);
    let rhs = Bn254::pairing(B_Q_x, srs.X2A[n] - srs.X2A[0]) + Bn254::pairing(corr3, srs.Y2A);
    assert!(lhs == rhs);

    //check B(x) - B(0) is divisible by x
    let lhs = Bn254::pairing(B_x - B_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(B_zero_div, srs.X2A[1]) + Bn254::pairing(corr4, srs.Y2A);
    assert!(lhs == rhs);

    // Sumcheck C
    let lhs = Bn254::pairing(inputs[2].g1, alpha_beta_pow_g2) - Bn254::pairing(C_x, srs.X2A[0]);
    let rhs = Bn254::pairing(C_Q_x, srs.X2A[2 * n] - srs.X2A[0]) + Bn254::pairing(corr5, srs.Y2A);
    assert!(lhs == rhs);

    // Assume C(0) = (A(0) + B(0)) / 2 (which assumes ∑A + ∑B = ∑C)
    let C_zero: G1Affine = ((A_zero + B_zero) * Fr::from(2).inverse().unwrap()).into();

    //check C(x) - C(0) is divisible by x
    let lhs = Bn254::pairing(C_x - C_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(C_zero_div, srs.X2A[1]) + Bn254::pairing(corr6, srs.Y2A);
    assert!(lhs == rhs);
  }
}
