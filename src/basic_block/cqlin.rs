#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use super::{BasicBlock, BasicBlockType, Data, DataEnc, SRS};
use crate::{
  setup::{CQLinSetup, CQSetup},
  util::{self, calc_pow},
};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::{One, UniformRand, Zero};
use ndarray::{ArrayD, Ix2};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

pub struct CQLinBasicBlock {
  pub weights_name: String,
}

// input is rows of A, model is rows of B, outputs are rows of C
impl BasicBlock for CQLinBasicBlock {
  fn block_type(&self) -> BasicBlockType {
    BasicBlockType::CQLin
  }

  fn name(&self) -> String {
    format!("CQLin-{}", self.weights_name)
  }

  fn weights_name(&self) -> String {
    self.weights_name.clone()
  }

  fn run(&self, weights: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(weights.ndim() == 2 && inputs.len() == 1 && inputs[0].ndim() == 2 && inputs[0].shape()[1] == weights.shape()[0]);
    let (a, b) = (
      weights.view().into_dimensionality::<Ix2>().unwrap(),
      inputs[0].view().into_dimensionality::<Ix2>().unwrap(),
    );
    vec![b.dot(&a).into_dyn()]
  }

  fn prove(
    &mut self,
    srs: &SRS,
    setup: &(Option<&CQLinSetup>, Option<&CQSetup>),
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let setup = setup.0.unwrap();
    let weights = &setup.weights;
    let l = inputs[0].len();
    let m = weights.len();
    let n = weights[0].raw.len();
    let N = srs.X2P.len() - 1;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let alpha = Fr::rand(rng);

    let alpha_pow = calc_pow(alpha, l);
    let alpha_pow = Data::new(srs, &alpha_pow); //r is ignored (unnecessary msm)

    let mut flat_A = vec![Fr::zero(); m];
    let mut flat_A_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        flat_A[j] += inputs[0][i].raw[j] * alpha_pow.raw[i];
      }
      flat_A_r += inputs[0][i].r * alpha_pow.raw[i];
    }
    let mut flat_A = Data::new(srs, &flat_A);
    flat_A.r = flat_A_r;

    let mut flat_C = vec![Fr::zero(); n];
    let mut flat_C_r = Fr::zero();
    for i in 0..l {
      for j in 0..n {
        flat_C[j] += outputs[0][i].raw[j] * alpha_pow.raw[i];
      }
      flat_C_r += outputs[0][i].r * alpha_pow.raw[i];
    }
    let mut flat_C = Data::new(srs, &flat_C);
    flat_C.r = flat_C_r;

    let R = &setup.R;
    let Q = &setup.Q;
    let S = &setup.S;
    let P_R = &setup.P_R;
    let L_V_i_x_n = &setup.L_V_i_x_n;
    let L_V_i_x = &setup.L_V_i_x;

    let R_x = util::msm::<G1Projective>(R, &flat_A.raw).into();
    let Q_x = util::msm::<G1Projective>(Q, &flat_A.raw).into();
    let temp: Vec<_> = (0..m).into_par_iter().map(|i| srs.X1A[n * i]).collect();
    let A_x = util::msm::<G1Projective>(&temp, &flat_A.poly.coeffs).into();
    let S_x = util::msm::<G1Projective>(S, &flat_A.raw).into();
    let P_x = util::msm::<G1Projective>(&srs.X1A[N - n..N], &flat_C.poly.coeffs).into();
    let P_R_x: G1Affine = util::msm::<G1Projective>(&P_R, &flat_A.raw).into();

    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let z = flat_A.poly.evaluate(&gamma_n);
    let h_i: Vec<_> = (0..m).into_par_iter().map(|i| (flat_A.raw[i] - z) * (domain_m.element(i) - gamma_n).inverse().unwrap()).collect();
    let z = (srs.X1P[0] * z).into();
    let pi = util::msm::<G1Projective>(&L_V_i_x, &h_i).into();
    let pi_1 = util::msm::<G1Projective>(&L_V_i_x_n, &h_i).into();

    let mut rng2 = StdRng::from_entropy();
    // R, Q, A, S, P, pR, pi, pi_1, M
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
    let proof = vec![R_x, Q_x, A_x, S_x, P_x, P_R_x, pi, pi_1];
    let mut proof: Vec<_> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    proof.push(z);

    // G1 M needed for blinding
    // h, h_S, h_g, h_R, h_pi, h_pi_1
    let M_x_1 = R.iter().sum::<G1Projective>();
    let mut C = vec![
      M_x_1 * r[2] - (srs.X1P[m * n] - srs.X1P[0]) * r[1] - srs.X1P[0] * r[0] + srs.Y1P * r[2] * r[8] + A_x * r[8],
      srs.X1P[0] * (r[0] - flat_C.r * Fr::from(m as u32).inverse().unwrap()) - srs.X1P[n] * r[3],
      srs.X1P[N - n] * flat_C.r - srs.X1P[0] * r[4],
      srs.X1P[N - m * n] * r[0] - srs.X1P[0] * r[5],
      srs.X1P[0] * flat_A.r + (srs.X1P[0] * gamma_n - srs.X1P[1]) * r[6],
      srs.X1P[0] * r[2] + (srs.X1P[0] * gamma_n - srs.X1P[n]) * r[7],
    ];

    proof.append(&mut C);

    // G2 blinding for M
    let M_x_2 = (setup.M_x + srs.Y2P * r[8]).into();

    return (proof, vec![M_x_2]);
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let l = inputs[0].len();
    let m = model.len();
    let n = model[0].len;
    let N = srs.X2P.len() - 1;

    let [R_x, Q_x, A_x, S_x, P_x, P_R_x, pi, pi_1, z, C1, C2, C3, C4, C5, C6] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [M_x] = proof.1[..] else { panic!("Wrong proof format") };

    let alpha = Fr::rand(rng);
    let alpha_pow = calc_pow(alpha, l);

    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| inputs[0][i].g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| outputs[0][i].g1).collect();
    let flat_C_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Check A(x) M(x) = Z(X) Q(X) + R(X)
    let lhs = Bn254::pairing(A_x, M_x);
    let rhs = Bn254::pairing(Q_x, srs.X2A[m * n] - srs.X2A[0]) + Bn254::pairing(R_x, srs.X2A[0]) + Bn254::pairing(C1, srs.Y2A);
    assert!(lhs == rhs);

    // Check R(X) - 1/m g(X) = S(X) X^n
    let temp: G1Affine = (flat_C_g1 * Fr::from(m as u64).inverse().unwrap()).into();
    let lhs = Bn254::pairing(R_x - temp, srs.X2A[0]);
    let rhs = Bn254::pairing(S_x, srs.X2A[n]) + Bn254::pairing(C2, srs.Y2A);
    assert!(lhs == rhs);

    // n degree-check for g
    let lhs = Bn254::pairing(flat_C_g1, srs.X2A[N - n]);
    let rhs = Bn254::pairing(P_x, srs.X2A[0]) + Bn254::pairing(C3, srs.Y2A);
    assert!(lhs == rhs);

    // mn degree-check for R
    let lhs = Bn254::pairing(R_x, srs.X2A[N - m * n]);
    let rhs = Bn254::pairing(P_R_x, srs.X2A[0]) + Bn254::pairing(C4, srs.Y2A);
    assert!(lhs == rhs);

    // Checks A(gamma) = f(gamma^n)
    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let lhs = Bn254::pairing(flat_A_g1 - z + pi * gamma_n, srs.X2A[0]);
    let rhs = Bn254::pairing(pi, srs.X2A[1]) + Bn254::pairing(C5, srs.Y2A);
    assert!(lhs == rhs);

    let lhs = Bn254::pairing(A_x - z + pi_1 * gamma_n, srs.X2A[0]);
    let rhs = Bn254::pairing(pi_1, srs.X2A[n]) + Bn254::pairing(C6, srs.Y2A);
    assert!(lhs == rhs);
  }
}
