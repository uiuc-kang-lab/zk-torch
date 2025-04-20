#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_to_acc_proof, calc_pow, AccHolder};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::{One, UniformRand, Zero};
use ndarray::{ArrayD, Ix2, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

fn acc_proof_to_cqlin_acc<P: Clone, Q: Clone>(acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>), log_n: usize, is_prover: bool) -> AccHolder<P, Q> {
  if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
    return AccHolder {
      acc_g1: vec![],
      acc_g2: vec![],
      acc_fr: vec![],
      mu: Fr::zero(),
      errs: vec![],
      acc_errs: vec![],
    };
  }

  let acc_g1_num = if is_prover { 20 } else { 17 };
  let acc_fr_num = if is_prover { log_n + 3 } else { log_n + 1 };
  let acc_err_g2_num = acc_proof.1.len() - 3;

  let err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[acc_g1_num..(acc_g1_num + 5)].to_vec(), acc_proof.1[1..3].to_vec(), vec![]);
  let err5: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[(acc_g1_num + 5)..(acc_g1_num + 8)].to_vec(), vec![], vec![]);
  let err6: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[(acc_g1_num + 8)..(acc_g1_num + 11)].to_vec(), vec![], vec![]);

  let mut errs = vec![err1, err5, err6];
  for i in 0..log_n {
    let err8i = (vec![], vec![], vec![acc_proof.2[acc_fr_num + i]]);
    errs.push(err8i);
  }

  let acc_err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 11)..(acc_g1_num + 14 + acc_err_g2_num)].to_vec(),
    acc_proof.1[3..(3 + acc_err_g2_num)].to_vec(),
    vec![],
  );
  let acc_err5: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 14 + acc_err_g2_num)..(acc_g1_num + 17 + acc_err_g2_num)].to_vec(),
    vec![],
    vec![],
  );
  let acc_err6: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 17 + acc_err_g2_num)..(acc_g1_num + 20 + acc_err_g2_num)].to_vec(),
    vec![],
    vec![],
  );

  let mut acc_errs = vec![acc_err1, acc_err5, acc_err6];
  for i in 0..log_n {
    let acc_err8i = (vec![], vec![], vec![acc_proof.2[acc_fr_num + log_n + i]]);
    acc_errs.push(acc_err8i);
  }

  AccHolder {
    acc_g1: acc_proof.0[..acc_g1_num].to_vec(),
    acc_g2: acc_proof.1[..1].to_vec(),
    acc_fr: acc_proof.2[..acc_fr_num].to_vec(),
    mu: acc_proof.2[acc_proof.2.len() - 1],
    errs,
    acc_errs,
  }
}

#[derive(Debug)]
pub struct CQLinBasicBlock {
  pub setup: ArrayD<Fr>,
}

// input is rows of A, model is rows of B, outputs are rows of C
impl BasicBlock for CQLinBasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    self.setup.clone()
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(model.ndim() == 2);
    assert!(inputs.len() == 1);
    // Support both 1d and 2d inputs
    assert!(
      (inputs[0].ndim() == 1 && inputs[0].shape()[0] == model.shape()[0]) || (inputs[0].ndim() == 2 && inputs[0].shape()[1] == model.shape()[0])
    );
    let b = if inputs[0].ndim() == 2 {
      inputs[0]
    } else {
      &inputs[0].clone().into_shape(IxDyn(&[1, inputs[0].len()])).unwrap()
    };
    let (a, b) = (
      model.view().into_dimensionality::<Ix2>().unwrap(),
      b.view().into_dimensionality::<Ix2>().unwrap(),
    );
    if inputs[0].ndim() == 1 {
      let prod = b.dot(&a).into_dyn();
      Ok(vec![prod.clone().into_shape(IxDyn(&[prod.shape()[1]])).unwrap()])
    } else {
      Ok(vec![b.dot(&a).into_dyn()])
    }
  }

  #[cfg(not(feature = "mock_prove"))]
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

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    let m = model.len();
    let n = model[0].raw.len();

    let mut L_H_i_x = srs.X1P[..n].to_vec();
    let mut L_V_i_x_n: Vec<_> = srs.X1P[..m].into_par_iter().map(|x| (*x).into()).collect();
    let mut L_V_i_x: Vec<G1Projective> = srs.X1P[..m].into_par_iter().map(|x| (*x).into()).collect();
    let R: Vec<_> = L_V_i_x.iter().map(|x| *x).collect();
    let mut Q: Vec<_> = L_V_i_x.iter().map(|x| *x).collect();
    let mut S: Vec<_> = L_V_i_x.iter().map(|x| *x).collect();
    let mut P_R: Vec<_> = L_V_i_x.iter().map(|x| *x).collect();

    let M_x = srs.X2P[0].clone();

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

    let input = if inputs[0].ndim() == 0 {
      &inputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &inputs[0]
    };
    let output = if outputs[0].ndim() == 0 {
      &outputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &outputs[0]
    };
    let mut flat_A = vec![Fr::zero(); m];
    let mut flat_A_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        flat_A[j] += input[i].raw[j] * alpha_pow.raw[i];
      }
      flat_A_r += input[i].r * alpha_pow.raw[i];
    }
    let mut flat_A = Data::new(srs, &flat_A);
    flat_A.r = flat_A_r;

    let mut flat_C = vec![Fr::zero(); n];
    let mut flat_C_r = Fr::zero();
    for i in 0..l {
      for j in 0..n {
        flat_C[j] += output[i].raw[j] * alpha_pow.raw[i];
      }
      flat_C_r += output[i].r * alpha_pow.raw[i];
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

    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let gamma = Fr::rand(rng);
    let mut gammas: Vec<Fr> = vec![gamma.clone()];
    for i in 0..log_n {
      gammas.push(gammas[i].pow(&[2]));
    }

    let gamma_n = gammas[log_n];
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
    let part_C1 = (srs.X1P[0] - srs.X1P[m * n]) * r[1] - srs.X1P[0] * r[0];
    let mut C = vec![
      M_x_1 * r[2] + part_C1 + srs.Y1P * r[2] * r[8] + A_x * r[8],
      srs.X1P[0] * (r[0] - flat_C.r * Fr::from(m as u32).inverse().unwrap()) - srs.X1P[n] * r[3],
      srs.X1P[N - n] * flat_C.r - srs.X1P[0] * r[4],
      srs.X1P[N - m * n] * r[0] - srs.X1P[0] * r[5],
      srs.X1P[0] * flat_A.r + (srs.X1P[0] * gamma_n - srs.X1P[1]) * r[6],
      srs.X1P[0] * r[2] + (srs.X1P[0] * gamma_n - srs.X1P[n]) * r[7],
    ];

    proof.append(&mut C);

    // G2 blinding for M
    let M_x_2 = (setup.1[0] + srs.Y2P * r[8]).into();

    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![flat_A.g1 + srs.Y1P * flat_A.r, flat_C.g1 + srs.Y1P * flat_C.r, part_C1, M_x_1, A_x.into()];

      proof.append(&mut additional_g1_for_acc);
      gammas.push(r[2]);
      gammas.push(r[8]);
    }

    return (proof, vec![M_x_2], gammas);
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
    let log_n = n.next_power_of_two().trailing_zeros() as usize;

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

    let input = if inputs[0].ndim() == 0 {
      &inputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &inputs[0]
    };
    let output = if outputs[0].ndim() == 0 {
      &outputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &outputs[0]
    };
    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| input[i].g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| output[i].g1).collect();
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
    let gammas: Vec<Fr> = proof.2.clone();
    let gamma_n = gammas[log_n];
    assert_eq!(gamma, gammas[0]);
    for i in 0..log_n {
      assert_eq!(gammas[i].pow(&[2]), gammas[i + 1]);
    }

    checks.push(vec![
      ((flat_A_g1 - z + pi * gamma_n).into(), srs.X2A[0]),
      (-pi, srs.X2A[1]),
      (-C5, srs.Y2A),
    ]);

    checks.push(vec![((A_x - z + pi_1 * gamma_n).into(), srs.X2A[0]), (-pi_1, srs.X2A[n]), (-C6, srs.Y2A)]);

    checks
  }

  fn acc_init(
    &self,
    _srs: &SRS,
    model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let n = model[0].raw.len();
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());
    let g1_zero = G1Projective::zero();
    let g2_zero = G2Projective::zero();
    let fr_zero = Fr::zero();

    let mut bytes = Vec::new();
    proof.0[..proof.0.len() - 5].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    proof.2[..proof.2.len() - 2].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);

    acc_proof.0.extend(vec![g1_zero; 11 * 2]);
    acc_proof.1.extend(vec![g2_zero; 2 * 2]);
    acc_proof.2.extend(vec![fr_zero; log_n * 2]);

    // mu
    acc_proof.2.push(Fr::one());

    acc_proof
  }

  // This function performs folding for the rest of the blocks in the computation
  fn acc_prove(
    &self,
    srs: &SRS,
    model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let n = model[0].raw.len();
    let log_n = n.next_power_of_two().trailing_zeros() as usize;

    let [R_x, Q_x, A_x, _S_x, _P_x, _P_R_x, pi, pi_1, z, _C1, _C2, _C3, _C4, C5, C6, flat_A, _flat_C, part_C1, M_x_1, A_x_1] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [M_x_2] = proof.1[..] else { panic!("Wrong proof format") };
    let beta_k = proof.2[log_n];

    let acc_holder = acc_proof_to_cqlin_acc(acc_proof, log_n, true);
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };

    let [acc_R, acc_Q, acc_A, _acc_S, _acc_P, _acc_P_R, acc_pi, acc_pi_1, acc_z, _acc_C1, _acc_C2, _acc_C3, _acc_C4, acc_C5, acc_C6, acc_flat_A, _acc_flat_C, acc_part_C1, acc_M_1, acc_A_1] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };
    let [acc_M] = acc_holder.acc_g2[..] else { panic!("Wrong proof format") };
    let acc_mu = acc_holder.mu;
    let acc_beta_k = acc_holder.acc_fr[log_n];

    let acc_mask_A = acc_holder.acc_fr[log_n + 1];
    let acc_mask_M = acc_holder.acc_fr[log_n + 2];
    let cqlin_mask_A = proof.2[log_n + 1];
    let cqlin_mask_M = proof.2[log_n + 2];

    // Compute error terms
    let err1: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) = (
      vec![
        A_x.into(),
        acc_A,
        acc_Q + Q_x * acc_mu,
        acc_R + R_x * acc_mu,
        acc_part_C1
          + part_C1 * acc_mu
          + acc_M_1 * cqlin_mask_A
          + M_x_1 * acc_mask_A
          + acc_A_1 * cqlin_mask_M
          + A_x_1 * acc_mask_M
          + srs.Y1P * (cqlin_mask_A * acc_mask_M + cqlin_mask_M * acc_mask_A),
      ],
      vec![acc_M, M_x_2.into()],
      vec![],
    );

    let err5: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) = (
      vec![
        acc_flat_A + flat_A * acc_mu - acc_z - z * acc_mu + acc_pi * beta_k + pi * acc_beta_k,
        acc_pi + pi * acc_mu,
        acc_C5 + C5 * acc_mu,
      ],
      vec![],
      vec![],
    );

    let err6 = (
      vec![
        acc_A + A_x * acc_mu - acc_z - z * acc_mu + acc_pi_1 * beta_k + pi_1 * acc_beta_k,
        acc_pi_1 + pi_1 * acc_mu,
        acc_C6 + C6 * acc_mu,
      ],
      vec![],
      vec![],
    );

    let mut err8s = vec![];
    for i in 0..log_n {
      let cqlin_beta_i = proof.2[i];
      let cqlin_beta_i_1 = proof.2[i + 1];
      let acc_beta_i = acc_proof.2[i];
      let acc_beta_i_1 = acc_proof.2[i + 1];
      let err = (
        vec![],
        vec![],
        vec![cqlin_beta_i * acc_beta_i + cqlin_beta_i * acc_beta_i - acc_beta_i_1 - cqlin_beta_i_1 * acc_mu],
      );
      err8s.push(err);
    }

    let mut errs = vec![err1.clone(), err5.clone(), err6.clone()];
    errs.extend(err8s.clone());

    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_holder.acc_g1[..acc_holder.acc_g1.len() - 11].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g1[acc_holder.acc_g1.len() - 5..acc_holder.acc_g1.len() - 3].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_fr[..acc_holder.acc_fr.len() - 2].serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..proof.0.len() - 5].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    proof.2[..proof.2.len() - 2].serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    // Random linear combination for folding (i.e., acc + cqlin * acc_gamma)
    new_acc_holder.acc_g1 = proof.0.iter().enumerate().map(|(i, x)| (*x * acc_gamma) + acc_proof.0[i]).collect();
    new_acc_holder.acc_g2 = vec![acc_M + M_x_2 * acc_gamma];
    new_acc_holder.acc_fr = proof.2.iter().enumerate().map(|(i, x)| *x * acc_gamma + acc_proof.2[i]).collect();
    new_acc_holder.mu = acc_mu + acc_gamma;
    new_acc_holder.errs = errs.clone();
    new_acc_holder.acc_errs = acc_holder.acc_errs;

    for i in 0..log_n + 3 {
      if i < 3 {
        errs[i].0 = errs[i].0.iter().map(|x| (*x * acc_gamma).into()).collect();
      } else {
        errs[i].2 = errs[i].2.iter().map(|x| *x * acc_gamma).collect();
      }
    }

    // Append error terms
    // err1
    let err1_g1_len = new_acc_holder.acc_errs[0].0.len();
    let q_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 3].clone() + errs[0].0[2];
    let r_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 2].clone() + errs[0].0[3];
    let c_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 1].clone() + errs[0].0[4];
    let mut errs_0_g1 = errs[0].0[..2].to_vec();
    let mut errs_0_g2 = errs[0].1[..2].to_vec();

    new_acc_holder.acc_errs[0].0 = new_acc_holder.acc_errs[0].0[..err1_g1_len - 3].to_vec();
    new_acc_holder.acc_errs[0].0.append(&mut errs_0_g1);
    new_acc_holder.acc_errs[0].0.push(q_term_g1);
    new_acc_holder.acc_errs[0].0.push(r_term_g1);
    new_acc_holder.acc_errs[0].0.push(c_term_g1);
    new_acc_holder.acc_errs[0].1.append(&mut errs_0_g2);

    // err5
    new_acc_holder.acc_errs[1].0.iter_mut().enumerate().for_each(|(i, x)| *x += errs[1].0[i]);

    // err6
    new_acc_holder.acc_errs[2].0.iter_mut().enumerate().for_each(|(i, x)| *x += errs[2].0[i]);

    // err8s
    for i in 3..log_n + 3 {
      new_acc_holder.acc_errs[i].2[0] += errs[i].2[0];
    }

    acc_to_acc_proof(new_acc_holder)
  }

  // This function cleans the blinding terms in accumulators for the verifier to do acc_verify without knowing them
  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    let n = self.setup.shape()[1];
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let mut acc_holder = acc_proof_to_cqlin_acc(acc_proof, log_n, true);
    // correct the blinding factor C1
    acc_holder.acc_g1[9] = acc_holder.acc_g1[acc_holder.acc_g1.len() - 3] * acc_holder.mu
      + acc_holder.acc_g1[acc_holder.acc_g1.len() - 2] * acc_holder.acc_fr[log_n + 1]
      + srs.Y1P * acc_holder.acc_fr[log_n + 1] * acc_holder.acc_fr[log_n + 2]
      + acc_holder.acc_g1[acc_holder.acc_g1.len() - 1] * acc_holder.acc_fr[log_n + 2];
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_holder.acc_g1[..acc_holder.acc_g1.len() - 3].to_vec();
    acc_holder.acc_fr = acc_holder.acc_fr[..acc_holder.acc_fr.len() - 2].to_vec();
    let acc_proof = acc_to_acc_proof(acc_holder);

    // remove blinding terms from bb proof for the verifier
    let cqlin_proof = (
      proof.0[..proof.0.len() - 5].to_vec(),
      proof.1.to_vec(),
      proof.2[..proof.2.len() - 2].to_vec(),
    );

    (
      (
        cqlin_proof.0.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.1.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.2,
      ),
      (
        acc_proof.0.iter().map(|x| (*x).into()).collect(),
        acc_proof.1.iter().map(|x| (*x).into()).collect(),
        acc_proof.2,
      ),
    )
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Option<bool> {
    let l = inputs[0].len();
    let n = model[0].len;
    let log_n = n.next_power_of_two().trailing_zeros() as usize;

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

    let prev_acc_holder = acc_proof_to_cqlin_acc(prev_acc_proof, log_n, false);
    let acc_holder = acc_proof_to_cqlin_acc(acc_proof, log_n, false);

    let gamma = Fr::rand(rng);
    let mut result = gamma == proof.2[0];
    for i in 0..log_n {
      result &= proof.2[i].pow(&[2]) == proof.2[i + 1];
    }

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0.serialize_uncompressed(&mut bytes).unwrap();
      proof.1.serialize_uncompressed(&mut bytes).unwrap();
      proof.2.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    let alpha_pow = calc_pow(alpha, l);

    let input = if inputs[0].ndim() == 0 {
      &inputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &inputs[0]
    };
    let output = if outputs[0].ndim() == 0 {
      &outputs[0].clone().into_shape(IxDyn(&[1])).unwrap()
    } else {
      &outputs[0]
    };
    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| input[i].g1).collect();
    let flat_A_g1: G1Projective = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| output[i].g1).collect();
    let flat_C_g1: G1Projective = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_holder.acc_g1[..prev_acc_holder.acc_g1.len() - 8].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g1[prev_acc_holder.acc_g1.len() - 2..].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_fr.serialize_uncompressed(&mut bytes).unwrap();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    proof.2.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let cqlin_proof_g1 = [
      R_x,
      Q_x,
      A_x,
      S_x,
      P_x,
      P_R_x,
      pi,
      pi_1,
      z,
      C1,
      C2,
      C3,
      C4,
      C5,
      C6,
      flat_A_g1.into(),
      flat_C_g1.into(),
    ];
    cqlin_proof_g1.iter().zip(prev_acc_holder.acc_g1.iter()).enumerate().for_each(|(i, (x, y))| {
      if i >= 9 && i < 15 {
        return; // No need to verify RLC for blinding factors
      }
      let z = *y + *x * acc_gamma;
      let z: G1Affine = z.into();
      result &= z == acc_holder.acc_g1[i];
    });
    result &= prev_acc_holder.acc_g2[0] + M_x * acc_gamma == acc_holder.acc_g2[0];
    proof.2.iter().zip(prev_acc_holder.acc_fr.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *y + *x * acc_gamma;
      result &= z == acc_holder.acc_fr[i];
    });
    result &= prev_acc_holder.mu + acc_gamma == acc_holder.mu;

    // Check RLC for errors
    for i in 0..log_n + 3 {
      if i < 3 {
        acc_holder.errs[i].0[acc_holder.errs[i].0.len() - 3..]
          .iter()
          .zip(prev_acc_holder.acc_errs[i].0[prev_acc_holder.acc_errs[i].0.len() - 3..].iter())
          .enumerate()
          .for_each(|(j, (x, y))| {
            let z = *y + *x * acc_gamma;
            result &= z == acc_holder.acc_errs[i].0[acc_holder.acc_errs[i].0.len() - 3 + j];
          });
      } else {
        let z = prev_acc_holder.acc_errs[i].2[0] + acc_holder.errs[i].2[0] * acc_gamma;
        result &= z == acc_holder.acc_errs[i].2[0];
      }
    }

    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let m = self.setup.shape()[0];
    let n = self.setup.shape()[1];
    let N = srs.X2P.len() - 1;
    let log_n = n.next_power_of_two().trailing_zeros() as usize;

    let acc_holder = acc_proof_to_cqlin_acc(acc_proof, log_n, false);

    let [acc_R, acc_Q, acc_A, acc_S, acc_P, acc_P_R, acc_pi, acc_pi_1, acc_z, acc_C1, acc_C2, acc_C3, acc_C4, acc_C5, acc_C6, acc_flat_A, acc_flat_C] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };
    let [acc_M] = acc_holder.acc_g2[..] else { panic!("Wrong proof format") };
    let acc_mu = acc_holder.mu;
    let acc_beta_k = acc_holder.acc_fr[log_n];
    let err_1 = &acc_holder.acc_errs[0];
    let err_5 = &acc_holder.acc_errs[1];
    let err_6 = &acc_holder.acc_errs[2];
    let err_8s = &acc_holder.acc_errs[3..];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], (srs.X2A[m * n] - srs.X2A[0]).into()));
    temp.push((err_1.0[err_1.1.len() + 1], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 2], srs.Y2A));

    let err_1 = temp;
    let err_5: PairingCheck = vec![(-err_5.0[0], srs.X2A[0]), (err_5.0[1], srs.X2A[1]), (err_5.0[2], srs.Y2A)];
    let err_6: PairingCheck = vec![(-err_6.0[0], srs.X2A[0]), (err_6.0[1], srs.X2A[n]), (err_6.0[2], srs.Y2A)];

    let mut acc_1: PairingCheck = vec![
      (acc_A, acc_M),
      ((-acc_Q * acc_mu).into(), (srs.X2A[m * n] - srs.X2A[0]).into()),
      ((-acc_R * acc_mu).into(), srs.X2A[0]),
      (-acc_C1, srs.Y2A),
    ];
    acc_1.extend(err_1);

    let g_m: G1Affine = (acc_flat_C * Fr::from(m as u64).inverse().unwrap()).into();
    let acc_2: PairingCheck = vec![((acc_R - g_m).into(), srs.X2A[0]), (-acc_S, srs.X2A[n]), (-acc_C2, srs.Y2A)];
    let acc_3: PairingCheck = vec![(acc_flat_C, srs.X2A[N - n]), (-acc_P, srs.X2A[0]), (-acc_C3, srs.Y2A)];
    let acc_4: PairingCheck = vec![(acc_R, srs.X2A[N - m * n]), (-acc_P_R, srs.X2A[0]), (-acc_C4, srs.Y2A)];
    let mut acc_5: PairingCheck = vec![
      (((acc_flat_A - acc_z) * acc_mu + acc_pi * acc_beta_k).into(), srs.X2A[0]),
      ((-acc_pi * acc_mu).into(), srs.X2A[1]),
      ((-acc_C5 * acc_mu).into(), srs.Y2A),
    ];
    acc_5.extend(err_5);

    let mut acc_6: PairingCheck = vec![
      (((acc_A - acc_z) * acc_mu + acc_pi_1 * acc_beta_k).into(), srs.X2A[0]),
      ((-acc_pi_1 * acc_mu).into(), srs.X2A[n]),
      ((-acc_C6 * acc_mu).into(), srs.Y2A),
    ];
    acc_6.extend(err_6);

    for i in 0..log_n {
      let acc_beta_i = acc_holder.acc_fr[i];
      let acc_beta_i_1 = acc_holder.acc_fr[i + 1];
      let err_8 = err_8s[i].2[0];
      let acc_8i = acc_beta_i * acc_beta_i - acc_beta_i_1 * acc_mu - err_8;
      assert!(acc_8i.is_zero());
    }

    let checks = vec![acc_1, acc_2, acc_3, acc_4, acc_5, acc_6];
    checks
  }

  fn acc_clean_errs(&self, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) {
    let n = self.setup.shape()[1];
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let mut acc_holder = acc_proof_to_cqlin_acc(acc_proof, log_n, false);
    acc_holder.errs = vec![];
    acc_to_acc_proof(acc_holder)
  }
}
