#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::{UniformRand, Zero};
use ndarray::ArrayD;
use rand::Rng;
use rayon::prelude::*;

pub struct CQLinBasicBlock;
impl BasicBlock for CQLinBasicBlock {
  fn run(model: &ArrayD<Fr>, inputs: &Vec<ArrayD<Fr>>) -> ArrayD<Fr> {
    assert_eq!(model.shape()[0], inputs[0].len());
    let m = model.shape()[0];
    let n = model.shape()[1];
    let mut r = ArrayD::zeros(vec![n]);
    for i in 0..n {
      for j in 0..m {
        r[i] += model[[j, i]] * inputs[0][j];
      }
    }
    return r;
  }
  fn setup(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let m = model.raw.shape()[0];
    let n = model.raw.shape()[1];
    let N = srs.1.len() - 1;
    let m_inv = Fr::from(m as u64).inverse().unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2m = GeneralEvaluationDomain::<Fr>::new(2 * m).unwrap();

    let srs_p: Vec<G1Projective> = srs.0[..m * n].into_par_iter().map(|x| (*x).into()).collect();
    let mut L_H_i_x = srs_p[..n].to_vec();
    util::ifft_in_place(domain_n, &mut L_H_i_x);
    let mut L_V_i_x_n: Vec<_> = (0..m).into_par_iter().map(|i| srs_p[n * i]).collect();
    util::ifft_in_place(domain_m, &mut L_V_i_x_n);
    let mut L_V_i_x: Vec<G1Projective> = srs_p[..m].into_par_iter().map(|x| (*x).into()).collect();
    util::ifft_in_place::<G1Projective>(domain_m, &mut L_V_i_x);

    // Calculate G1 U for R and S
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs_p[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate U for mn-degree check
    let srs_p_last_mn: Vec<G1Projective> = srs.0[N - m * n..N].into_par_iter().map(|x| (*x).into()).collect();
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs_p_last_mn[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U_P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U_P_R.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate G2 U for M
    let mut temp: Vec<Vec<G2Projective>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.1[i + n * j].into()).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U2: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U2.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut V = srs_p[m * n - n..m * n].to_vec();
    util::ifft_in_place(domain_n, &mut V);
    V.par_iter_mut().for_each(|x| *x *= m_inv);

    let mut srs_star: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| srs_p[n * i..n * i + n].to_vec()).collect();
    srs_star.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    srs_star = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs_star[m - 1 - j][i]).collect()).collect();
    srs_star.par_iter_mut().for_each(|x| x.append(&mut vec![G1Projective::zero(); m]));
    srs_star.par_iter_mut().for_each(|x| util::fft_in_place(domain_2m, x));

    let S: Vec<Vec<_>> = (0..m)
      .into_par_iter()
      .map(|i| (0..n).map(|j| (U[i][j] * domain_m.element(i).inverse().unwrap() - V[j]) * model.raw[[i, j]]).collect())
      .collect();
    let S: Vec<_> = S.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    let R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U[i][j] * model.raw[[i, j]]).collect()).collect();
    let R: Vec<_> = R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    let P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U_P_R[i][j] * model.raw[[i, j]]).collect()).collect();
    let P_R: Vec<_> = P_R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    // Calculate C for Q
    let mut C: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| model.raw[[j, i]]).collect()).collect();
    C.par_iter_mut().for_each(|x| domain_m.ifft_in_place(x));

    // Calculate Q. The C above corresponds to the C in the cqlin paper
    C.par_iter_mut().for_each(|x| x.append(&mut vec![Fr::zero(); m]));
    C.par_iter_mut().for_each(|x| domain_2m.fft_in_place(x));
    let temp: Vec<Vec<_>> = (0..2 * m).into_par_iter().map(|i| (0..n).map(|j| srs_star[j][i] * C[j][i]).collect()).collect();
    let mut temp: Vec<_> = temp.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    util::ifft_in_place(domain_2m, &mut temp);
    let mut temp = temp[m..].to_vec();
    util::fft_in_place(domain_m, &mut temp);
    let Q: Vec<_> = (0..m).into_par_iter().map(|i| temp[i] * domain_m.element(i) * m_inv).collect();
    let M_x = (0..m).into_par_iter().map(|i| (0..n).map(|j| U2[i][j] * model.raw[[i, j]]).sum::<G2Projective>()).sum::<G2Projective>(); //TODO: Change to msm

    let R: Vec<G1Affine> = R.par_iter().map(|x| (*x).into()).collect();
    let mut Q: Vec<G1Affine> = Q.par_iter().map(|x| (*x).into()).collect();
    let mut S: Vec<G1Affine> = S.par_iter().map(|x| (*x).into()).collect();
    let mut P_R: Vec<G1Affine> = P_R.par_iter().map(|x| (*x).into()).collect();
    let mut L_H_i_x: Vec<G1Affine> = L_H_i_x.par_iter().map(|x| (*x).into()).collect();
    let mut L_V_i_x: Vec<G1Affine> = L_V_i_x.into_par_iter().map(|x| x.into()).collect();
    let mut L_V_i_x_n: Vec<G1Affine> = L_V_i_x_n.par_iter().map(|x| (*x).into()).collect();

    let mut setup = R;
    setup.append(&mut Q);
    setup.append(&mut S);
    setup.append(&mut P_R);
    setup.append(&mut L_V_i_x_n);
    setup.append(&mut L_V_i_x);
    setup.append(&mut L_H_i_x);
    (setup, vec![M_x.into()])
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Data,
    inputs: &Vec<Data>,
    output: &Data,
    rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let m = model.raw.shape()[0];
    let n = model.raw.shape()[1];
    let N = srs.1.len() - 1;
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_mn = GeneralEvaluationDomain::<Fr>::new(m * n).unwrap();

    let R = &setup.0[..m];
    let Q = &setup.0[m..2 * m];
    let S = &setup.0[2 * m..3 * m];
    let P_R = &setup.0[3 * m..4 * m];
    let L_V_i_x_n = &setup.0[4 * m..5 * m];
    let L_V_i_x = &setup.0[5 * m..6 * m];
    let L_H_i_x = &setup.0[6 * m..];

    let v = inputs[0].raw.clone().into_raw_vec();
    let R_x = util::msm::<G1Projective>(R, &v).into();
    let Q_x = util::msm::<G1Projective>(Q, &v).into();
    let temp: Vec<_> = (0..n).into_par_iter().map(|i| srs.0[n * i]).collect();
    let A_x = util::msm::<G1Projective>(&temp, &inputs[0].poly.coeffs).into();
    let S_x = util::msm::<G1Projective>(S, &v).into();
    let P_x = util::msm::<G1Projective>(&srs.0[N - n..N], &output.poly.coeffs).into();
    let P_R_x: G1Affine = util::msm::<G1Projective>(&P_R, &v).into();

    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let z = inputs[0].poly.evaluate(&gamma_n);
    let h_i: Vec<_> = (0..m).into_par_iter().map(|i| (inputs[0].raw[i] - z) * (domain_m.element(i) - gamma_n).inverse().unwrap()).collect();
    let z = (srs.0[0] * z).into();
    let pi = util::msm::<G1Projective>(&L_V_i_x, &h_i).into();
    let pi_1 = util::msm::<G1Projective>(&L_V_i_x_n, &h_i).into();

    // TODO: Implement blinding
    let C0 = (srs.0[srs.1.len() - 1] * inputs[0].r).into();
    let C1 = (srs.0[srs.1.len() - 1] * output.r).into();

    return (vec![R_x, Q_x, A_x, S_x, P_x, P_R_x, pi, pi_1, z, C0, C1], vec![setup.1[0]]);
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut R,
  ) {
    let m = model.shape[0];
    let n = model.shape[1];
    let N = srs.1.len() - 1;

    let [R_x, Q_x, A_x, S_x, P_x, P_R_x, pi, pi_1, z, C0, C1] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    // Check A(x) M(x) = Z(X) Q(X) + R(X)
    let lhs = Bn254::pairing(A_x, proof.1[0]);
    let rhs = Bn254::pairing(Q_x, srs.1[m * n] - srs.1[0]) + Bn254::pairing(R_x, srs.1[0]);
    assert!(lhs == rhs);

    // Check R(X) - 1/m g(X) = S(X) X^n
    let temp: G1Affine = ((output.g1 - C1) * Fr::from(m as u64).inverse().unwrap()).into();
    let lhs = Bn254::pairing(R_x - temp, srs.1[0]);
    let rhs = Bn254::pairing(S_x, srs.1[n]);
    assert!(lhs == rhs);

    // n degree-check for g
    let lhs = Bn254::pairing(output.g1 - C1, srs.1[N - n]);
    let rhs = Bn254::pairing(P_x, srs.1[0]);
    assert!(lhs == rhs);

    // mn degree-check for R
    let lhs = Bn254::pairing(R_x, srs.1[N - m * n]);
    let rhs = Bn254::pairing(P_R_x, srs.1[0]);
    assert!(lhs == rhs);

    // Checks A(gamma) = f(gamma^n)
    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let lhs = Bn254::pairing(inputs[0].g1 - C0 - z + pi * gamma_n, srs.1[0]);
    let rhs = Bn254::pairing(pi, srs.1[1]);
    assert!(lhs == rhs);

    let lhs = Bn254::pairing(A_x - z + pi_1 * gamma_n, srs.1[0]);
    let rhs = Bn254::pairing(pi_1, srs.1[n]);
    assert!(lhs == rhs);
  }
}
