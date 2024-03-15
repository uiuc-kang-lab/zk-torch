#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use rand::{rngs::StdRng, SeedableRng};

struct LeftProof {
  x: G1Affine,
  Q_x: G1Affine,
  zero: G1Affine,
  zero_div: G1Affine,
}
struct RightProof {
  x: G1Affine,
  Q_x: G1Affine,
  zero_div: G1Affine,
}

fn calc_pow(alpha: Fr, n: usize) -> Vec<Fr>{
  let mut pow: Vec<Fr> = vec![Fr::one(); n];
  for i in 0..n - 1 {
    pow[i + 1] = pow[i] * alpha;
  }
  pow
}

pub struct MatMulBasicBlock{
  pub l: usize,
}
// inputs are rows to A and columns of B, outputs are rows of C
impl BasicBlock for MatMulBasicBlock {
  fn run(&self, _model: &Vec<Fr>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let l = self.l;
    let m = inputs[0].len();
    let n = inputs.len() - l;
    let mut r = vec![vec![Fr::zero(); n]; l];
    for i in 0..l {
      for j in 0..n {
        for k in 0..m {
          r[i][j] += inputs[i][k] * inputs[l+j][k];
        }
      }
    }
    r
  }
  fn prove(
    &mut self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let l = self.l;
    let m = inputs[0].raw.len();
    let n = inputs.len() - l;
    let domain_l = GeneralEvaluationDomain::<Fr>::new(l).unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, l);
    let alpha_pow = Data::new(srs, &alpha_pow); //r is ignored (unnecessary msm)
    let beta_pow = calc_pow(beta, n);
    let beta_pow = Data::new(srs, &beta_pow); //r is ignored

    let mut flat_A = vec![Fr::zero(); m];
    let mut flat_A_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        flat_A[j] += inputs[i].raw[j] * alpha_pow.raw[i];
      }
      flat_A_r += inputs[i].r * alpha_pow.raw[i];
    }
    let mut flat_A = Data::new(srs, &flat_A);
    flat_A.r = flat_A_r;

    let mut flat_B = vec![Fr::zero(); m];
    let mut flat_B_r = Fr::zero();
    for i in 0..n {
      for j in 0..m {
        flat_B[j] += inputs[l+i].raw[j] * beta_pow.raw[i];
      }
      flat_B_r += inputs[l+i].r * beta_pow.raw[i];
    }
    let mut flat_B = Data::new(srs, &flat_B);
    flat_B.r = flat_B_r;

    let mut flat_C = vec![Fr::zero(); n];
    let mut flat_C_r = Fr::zero();
    for i in 0..l {
      for j in 0..n {
        flat_C[j] += outputs[i].raw[j] * alpha_pow.raw[i];
      }
      flat_C_r += outputs[i].r * alpha_pow.raw[i];
    }
    let mut flat_C = Data::new(srs, &flat_C);
    flat_C.r = flat_C_r;

    // Calculate Left
    let left_raw: Vec<Fr> = (0..m).map(|i| flat_A.raw[i] * flat_B.raw[i]).collect();
    let left_poly = DensePolynomial { coeffs: domain_m.ifft(&left_raw) };
    let left_Q_poly = flat_A.poly.mul(&flat_B.poly).sub(&left_poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let left = LeftProof {
      x: util::msm::<G1Projective>(&srs.0, &left_poly.coeffs).into(),
      Q_x: util::msm::<G1Projective>(&srs.0, &left_Q_poly.coeffs).into(),
      zero: (srs.0[0] * (Fr::from(m as u32).inverse().unwrap() * left_raw.iter().sum::<Fr>())).into(),
      zero_div: util::msm::<G1Projective>(&srs.0, &left_poly.coeffs[1..]).into(),
    };
    let flat_B_g2 = util::msm::<G2Projective>(&srs.1, &flat_B.poly.coeffs) + srs.1[srs.1.len() - 1] * flat_B.r;
    let flat_B_g2 = flat_B_g2.into();

    // Calculate Right
    let right_raw: Vec<Fr> = (0..n).map(|i| flat_C.raw[i] * beta_pow.raw[i]).collect();
    let right_poly = DensePolynomial { coeffs: domain_n.ifft(&right_raw) };
    let right_Q_poly = flat_C.poly.mul(&beta_pow.poly).sub(&right_poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let right = RightProof {
      x: util::msm::<G1Projective>(&srs.0, &right_poly.coeffs).into(),
      Q_x: util::msm::<G1Projective>(&srs.0, &right_Q_poly.coeffs).into(),
      zero_div: util::msm::<G1Projective>(&srs.0, &right_poly.coeffs[1..]).into(),
    };

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..7).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Affine> = vec![left.x, left.Q_x, left.zero, left.zero_div, right.x, right.Q_x, right.zero_div];
    let mut proof: Vec<G1Affine> = proof.iter().enumerate().map(|(i, x)| ((*x) + srs.0[srs.1.len() - 1] * r[i]).into()).collect();
    let corr = vec![
      -(srs.0[m] - srs.0[0]) * r[1] + flat_A.g1 * flat_B.r + flat_B.g1 * flat_A.r + srs.0[srs.1.len() - 1] * flat_A.r * flat_B.r - srs.0[0] * r[0],
      -srs.0[1] * r[3] + srs.0[0] * (r[0] - r[2]),
      -(srs.0[n] - srs.0[0]) * r[5] + beta_pow.g1 * flat_C.r - srs.0[0] * r[4],
      -srs.0[1] * r[6] + srs.0[0] * (r[4] - r[2] * Fr::from(m as u32) * Fr::from(n as u32).inverse().unwrap()),
    ];
    let mut corr: Vec<G1Affine> = corr.iter().map(|x| (*x).into()).collect();
    proof.append(&mut corr);

    return (proof, vec![flat_B_g2]);
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let l = self.l;
    let m = inputs[0].len;
    let n = inputs.len() - l;
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let left = LeftProof {
      x: proof.0[0],
      Q_x: proof.0[1],
      zero: proof.0[2],
      zero_div: proof.0[3],
    };
    let right = RightProof {
      x: proof.0[4],
      Q_x: proof.0[5],
      zero_div: proof.0[6],
    };
    let [corr1, corr2, corr3, corr4] = proof.0[7..] else { panic!("Wrong proof format") };
    let flat_B_g2 = proof.1[0];

    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, l); //r is ignored
    let beta_pow = calc_pow(beta, n); //r is ignored
    let beta_pow_coeff = domain_n.ifft(&beta_pow);
    let beta_pow_g2: G2Affine = util::msm::<G2Projective>(&srs.1, &beta_pow_coeff).into();

    // Calculate flat_A
    let temp:Vec<_> = (0..l).map(|i|inputs[i].g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_B
    let temp:Vec<_> = (0..n).map(|i|inputs[l+i].g1).collect();
    let flat_B_g1 = util::msm::<G1Projective>(&temp, &beta_pow);

    // Calculate flat_C
    let temp:Vec<_> = (0..l).map(|i|outputs[i].g1).collect();
    let flat_C_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Check left(x) (left_i = flat_A_i * flat_B_i)
    let lhs = Bn254::pairing(flat_A_g1, flat_B_g2) - Bn254::pairing(left.x, srs.1[0]);
    let rhs = Bn254::pairing(left.Q_x, srs.1[m] - srs.1[0]) + Bn254::pairing(corr1, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check flat_B_g2
    let lhs = Bn254::pairing(flat_B_g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], flat_B_g2);
    assert!(lhs == rhs);

    // Check left(x) - left(0) is divisible by x
    let lhs = Bn254::pairing(left.x - left.zero, srs.1[0]);
    let rhs = Bn254::pairing(left.zero_div, srs.1[1]) + Bn254::pairing(corr2, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check right(x) (right_i = flat_C_i * beta_pow_i)
    let lhs = Bn254::pairing(flat_C_g1, beta_pow_g2) - Bn254::pairing(right.x, srs.1[0]);
    let rhs = Bn254::pairing(right.Q_x, srs.1[n] - srs.1[0]) + Bn254::pairing(corr3, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Assume right(0) = left(0)*n/m (which assumes ∑left=∑right)
    let right_zero: G1Affine = (left.zero * (Fr::from(m as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    //check right(x) - right(0) is divisible by x
    let lhs = Bn254::pairing(right.x - right_zero, srs.1[0]);
    let rhs = Bn254::pairing(right.zero_div, srs.1[1]) + Bn254::pairing(corr4, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
  }
}
