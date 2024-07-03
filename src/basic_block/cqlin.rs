#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, calc_pow};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::{UniformRand, Zero};
use ndarray::{ArrayD, Ix2};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug)]
pub struct CQLinBasicBlock;
// input is rows of A, model is rows of B, outputs are rows of C
impl BasicBlock for CQLinBasicBlock {
  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(model.ndim() == 2 && inputs.len() == 1 && inputs[0].ndim() == 2 && inputs[0].shape()[1] == model.shape()[0]);
    let (a, b) = (
      model.view().into_dimensionality::<Ix2>().unwrap(),
      inputs[0].view().into_dimensionality::<Ix2>().unwrap(),
    );
    vec![b.dot(&a).into_dyn()]
  }

  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let m = model.len();
    let n = model[0].raw.len();
    let N = srs.X2P.len() - 1;
    let m_inv = Fr::from(m as u64).inverse().unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2m = GeneralEvaluationDomain::<Fr>::new(2 * m).unwrap();

    let mut L_H_i_x = srs.X1P[..n].to_vec();
    util::ifft_in_place(domain_n, &mut L_H_i_x);
    let mut L_V_i_x_n: Vec<_> = (0..m).into_par_iter().map(|i| srs.X1P[n * i]).collect();
    util::ifft_in_place(domain_m, &mut L_V_i_x_n);
    let mut L_V_i_x: Vec<G1Projective> = srs.X1P[..m].into_par_iter().map(|x| (*x).into()).collect();
    util::ifft_in_place::<G1Projective>(domain_m, &mut L_V_i_x);

    // Calculate G1 U for R and S
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X1P[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate U for mn-degree check
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X1P[N - m * n + i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U_P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U_P_R.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate G2 U for M
    let mut temp: Vec<Vec<G2Projective>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X2P[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U2: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U2.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut V = srs.X1P[m * n - n..m * n].to_vec();
    util::ifft_in_place(domain_n, &mut V);
    V.par_iter_mut().for_each(|x| *x *= m_inv);

    let mut srs_star: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| srs.X1P[n * i..n * i + n].to_vec()).collect();
    srs_star.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    srs_star = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs_star[m - 1 - j][i]).collect()).collect();
    srs_star.par_iter_mut().for_each(|x| x.append(&mut vec![G1Projective::zero(); m]));
    srs_star.par_iter_mut().for_each(|x| util::fft_in_place(domain_2m, x));

    let S: Vec<Vec<_>> = (0..m)
      .into_par_iter()
      .map(|i| (0..n).map(|j| (U[i][j] * domain_m.element(i).inverse().unwrap() - V[j]) * model[i].raw[j]).collect())
      .collect();
    let mut S: Vec<_> = S.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    let R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U[i][j] * model[i].raw[j]).collect()).collect();
    let R: Vec<_> = R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    let P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U_P_R[i][j] * model[i].raw[j]).collect()).collect();
    let mut P_R: Vec<_> = P_R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    // Calculate C for Q
    let mut C: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| model[j].raw[i]).collect()).collect();
    C.par_iter_mut().for_each(|x| domain_m.ifft_in_place(x));

    // Calculate Q. The C above corresponds to the C in the cqlin paper
    C.par_iter_mut().for_each(|x| x.append(&mut vec![Fr::zero(); m]));
    C.par_iter_mut().for_each(|x| domain_2m.fft_in_place(x));
    let temp: Vec<Vec<_>> = (0..2 * m).into_par_iter().map(|i| (0..n).map(|j| srs_star[j][i] * C[j][i]).collect()).collect();
    let mut temp: Vec<_> = temp.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    util::ifft_in_place(domain_2m, &mut temp);
    let mut temp = temp[m..].to_vec();
    util::fft_in_place(domain_m, &mut temp);
    let mut Q: Vec<_> = (0..m).into_par_iter().map(|i| temp[i] * domain_m.element(i) * m_inv).collect();
    let M_x = (0..m).into_par_iter().map(|i| (0..n).map(|j| U2[i][j] * model[i].raw[j]).sum::<G2Projective>()).sum::<G2Projective>(); // TODO: Change to msm

    let mut setup = R;
    setup.append(&mut Q);
    setup.append(&mut S);
    setup.append(&mut P_R);
    setup.append(&mut L_V_i_x_n);
    setup.append(&mut L_V_i_x);
    setup.append(&mut L_H_i_x);
    (setup, vec![M_x.into()], Vec::new())
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let l = inputs[0].len();
    let m = model.len();
    let n = model[0].raw.len();
    let N = srs.X2P.len() - 1;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("cqlin_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
    };

    let alpha_pow = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::Data(alpha_pow) =
        cache.entry(format!("cqlin_alpha_msm_{l}")).or_insert_with(|| CacheValues::Data(Data::new(srs, &calc_pow(alpha, l))))
      else {
        panic!("Cache type error")
      };
      alpha_pow.clone()
    };

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

    let R = &setup.0[..m];
    let Q = &setup.0[m..2 * m];
    let S = &setup.0[2 * m..3 * m];
    let P_R = &setup.0[3 * m..4 * m];
    let L_V_i_x_n = &setup.0[4 * m..5 * m];
    let L_V_i_x = &setup.0[5 * m..6 * m];

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
    let M_x_2 = (setup.1[0] + srs.Y2P * r[8]).into();

    return (proof, vec![M_x_2], Vec::new());
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let l = inputs[0].len();
    let m = model.len();
    let n = model[0].len;
    let N = srs.X2P.len() - 1;

    let [R_x, Q_x, A_x, S_x, P_x, P_R_x, pi, pi_1, z, C1, C2, C3, C4, C5, C6] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [M_x] = proof.1[..] else { panic!("Wrong proof format") };

    let alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("cqlin_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
    };
    let alpha_pow = calc_pow(alpha, l);

    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| inputs[0][i].g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| outputs[0][i].g1).collect();
    let flat_C_g1 = util::msm::<G1Projective>(&temp, &alpha_pow).into();

    // Check A(x) M(x) = Z(X) Q(X) + R(X)
    checks.push(vec![
      (A_x, M_x),
      (-Q_x, (srs.X2A[m * n] - srs.X2A[0]).into()),
      (-R_x, srs.X2A[0]),
      (-C1, srs.Y2A),
    ]);

    // Check R(X) - 1/m g(X) = S(X) X^n
    let temp: G1Projective = flat_C_g1 * Fr::from(m as u64).inverse().unwrap();
    let temp: G1Affine = temp.into();
    checks.push(vec![((R_x - temp).into(), srs.X2A[0]), (-S_x, srs.X2A[n]), (-C2, srs.Y2A)]);

    // n degree-check for g
    checks.push(vec![(flat_C_g1, srs.X2A[N - n]), (-P_x, srs.X2A[0]), (-C3, srs.Y2A)]);

    // mn degree-check for R
    checks.push(vec![(R_x, srs.X2A[N - m * n]), (-P_R_x, srs.X2A[0]), (-C4, srs.Y2A)]);

    // Checks A(gamma) = f(gamma^n)
    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    checks.push(vec![
      ((flat_A_g1 - z + pi * gamma_n).into(), srs.X2A[0]),
      (-pi, srs.X2A[1]),
      (-C5, srs.Y2A),
    ]);

    checks.push(vec![((A_x - z + pi_1 * gamma_n).into(), srs.X2A[0]), (-pi_1, srs.X2A[n]), (-C6, srs.Y2A)]);

    checks
  }
}
