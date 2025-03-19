#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_to_acc_proof, AccHolder};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{
  ops::{Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::{Array1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub fn acc_proof_to_cq_acc<P: Clone, Q: Clone>(acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>), is_prover: bool) -> AccHolder<P, Q> {
  // If the proof is empty, return an empty AccHolder
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

  // Calculate sizes based on whether this is for prover or verifier
  let acc_g1_num = if is_prover { 20 } else { 14 }; // Main proof elements
  let acc_g2_num = 2;
  let acc_fr_num = if is_prover { 5 } else { 1 };
  // let acc_err_g2_num = acc_proof.1.len() - acc_g2_num;

  // Extract error terms

  // m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5
  // err_1: r⋅(A'(x)T(x)+A(x)T'(x)-μ⋅Q_A(x)Z_V(X)-Q'_A(x)Z_V(X)+μ⋅M(x)+M'(x)-βA'(x)-β'A(x))
  let err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[acc_g1_num..(acc_g1_num + 7)].to_vec(), // [A_x, acc_A_x, acc_A_Q_x + A_Q_x * acc_mu, m_x * acc_mu + acc_m_x]
    acc_proof.1[2..4].to_vec(),                         // [T_x_2, acc_T_x_2]
    vec![],
  );

  // err_4: r⋅(B'(x)f(x)+B(x)f'(x)-μ⋅Q_B(x)Z_H(X)-Q'_B(x)Z_H(X)+μ⋅B(x)+B'(x)-2μ)
  let err4: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 7)..(acc_g1_num + 12)].to_vec(), // [B_x, acc_B_x, B_Q_x, acc_B_Q_x]
    acc_proof.1[4..6].to_vec(),                                // [f_x_2, acc_f_x_2]
    vec![acc_proof.2[acc_fr_num]],                             // -2μ term
  );

  // Create vector of error terms
  let errs = vec![err1, err4];

  // Extract accumulated error terms
  let acc_err_offset = acc_g1_num + 12;
  let acc_g2_offset = 6;

  // let err_1 = (
  //   vec![A_x, acc_A_x, acc_A_Q_x + A_Q_x * acc_mu, m_x * acc_mu + acc_m_x],
  //   vec![T_x_2, acc_T_x_2],
  //   vec![],
  // );

  // // err_4: r⋅(B'(x)f(x)+B(x)f'(x)-μ⋅Q_B(x)Z_H(X)-Q'_B(x)Z_H(X)+μ⋅B(x)+B'(x)-2μ)
  // let err_4 = (
  //   vec![B_x, acc_B_x, acc_B_Q_x + B_Q_x * acc_mu, B_x * acc_mu + acc_B_x],
  //   vec![f_x_2, acc_f_x_2],
  //   vec![-Fr::from(2) * acc_mu],
  // );

  // acc_err1
  let acc_err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[acc_err_offset..(acc_err_offset + 7)].to_vec(),
    acc_proof.1[acc_g2_offset..(acc_g2_offset + 2)].to_vec(),
    vec![],
  );

  // acc_err4
  let acc_err4: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_err_offset + 7)..(acc_err_offset + 12)].to_vec(),
    acc_proof.1[acc_g2_offset + 2..(acc_g2_offset + 4)].to_vec(),
    vec![acc_proof.2[acc_proof.2.len() - 2]],
  );

  let acc_errs = vec![acc_err1, acc_err4];

  // Return structured AccHolder
  AccHolder {
    acc_g1: acc_proof.0[..acc_g1_num].to_vec(),
    acc_g2: acc_proof.1[..acc_g2_num].to_vec(),
    acc_fr: acc_proof.2[..acc_fr_num].to_vec(),
    mu: acc_proof.2[acc_proof.2.len() - 1],
    errs,
    acc_errs,
  }
}

#[derive(Debug)]
pub struct CQBasicBlock {
  pub n: usize,
  pub setup: util::CQArrayType,
}

impl BasicBlock for CQBasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    util::gen_cq_array(self.setup.clone())
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    if model.len() == 0 {
      return Ok(vec![]);
    }
    assert!(inputs.len() == 1);
    for x in inputs[0].view().as_slice().unwrap() {
      let x_int = util::fr_to_int(*x);
      if !util::check_cq_array(self.setup.clone(), x_int) {
        return Err(util::CQOutOfRangeError { input: x_int });
      }
    }
    Ok(vec![])
  }

  #[cfg(not(feature = "mock_prove"))]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    assert!(model.len() == 1);
    let model = &model.first().unwrap();
    let N = model.raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let T_x_2 = util::msm::<G2Projective>(&srs.X2A, &model.poly.coeffs) + srs.Y2P * model.r;
    let mut temp = model.poly.coeffs[1..].to_vec();
    temp.resize(N * 2 - 1, Fr::zero());
    let mut temp2 = srs.X1P[..N].to_vec();
    temp2.reverse();
    let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
    util::fft_in_place(domain_N, &mut Q_i_x_1);
    let temp = Fr::from(N as u32).inverse().unwrap();
    let temp2 = domain_N.group_gen_inv().pow(&[(N - 1) as u64]);
    let scalars: Vec<_> = (0..N).into_par_iter().map(|i| temp * temp2.pow(&[i as u64])).collect();
    util::ssm_g1_in_place(&mut Q_i_x_1, &scalars);
    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let scalars = (0..N).into_par_iter().map(|i| domain_N.group_gen_inv().pow(&[i as u64])).collect();
    util::ssm_g1_in_place(&mut L_i_0_x_1, &scalars);

    let temp = srs.X1P[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().for_each(|x| *x -= temp);

    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![T_x_2], Vec::new());
  }

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    assert!(model.len() == 1);
    let model = &model.first().unwrap();
    let N = model.raw.len();
    let L_i_x_1 = srs.X1P[..N].to_vec();
    let L_i_0_x_1 = L_i_x_1.clone();
    let Q_i_x_1 = L_i_x_1.clone();

    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![srs.X2P[0]], Vec::new());
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    assert!(inputs.len() == 1 && inputs[0].len() == 1);
    let model = &model.first().unwrap();
    let input = &inputs[0].first().unwrap();
    let N = model.raw.len();
    let n = input.raw.len();
    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    // gen(N, t):
    let Q_i_x_1 = &setup.0[..N];
    let L_i_x_1 = &setup.0[N..2 * N];
    let L_i_0_x_1 = &setup.0[2 * N..];
    let m_i = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::CQTableDict(table_dict) =
        cache.entry(format!("cq_table_dict_{:p}", self)).or_insert_with(|| CacheValues::CQTableDict(HashMap::new()))
      else {
        panic!("Cache type error")
      };
      if table_dict.len() == 0 {
        for i in 0..N {
          table_dict.insert(model.raw[i], i);
        }
      }

      // Calculate m
      let mut m_i = HashMap::new();
      for x in input.raw.iter() {
        if !table_dict.contains_key(x) {
          println!("{:?},{:?}", x, -*x);
        }
        m_i.entry(table_dict.get(x).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
      }
      m_i
    };
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[*i], Fr::from(*y as u32))).unzip();
    let m_x = util::msm::<G1Projective>(&temp, &temp2);

    let beta = Fr::rand(rng);

    // Calculate A
    let A_i: HashMap<usize, Fr> = m_i.iter().map(|(i, y)| (*i, Fr::from(*y as u32) * (model.raw[*i] + beta).inverse().unwrap())).collect();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
    let A_x = util::msm::<G1Projective>(&temp, &temp2);
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
    let A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
    let A_zero = srs.X1P[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>());
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
    let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);

    // Calculate B
    let B_i: Vec<Fr> = input.raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
    let B_poly = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_Q_poly = B_poly
      .mul(&(input.poly.clone() + (DensePolynomial::from_coefficients_vec(vec![beta]))))
      .sub(&DensePolynomial::from_coefficients_vec(vec![Fr::one()]))
      .divide_by_vanishing_poly(domain_n)
      .unwrap()
      .0;
    let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
    let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
    let B_zero_div = if B_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..])
    };
    let B_DC = util::msm::<G1Projective>(&srs.X1A[N - n..], &B_poly.coeffs);

    let f_x_2 = util::msm::<G2Projective>(&srs.X2A, &input.poly.coeffs) + srs.Y2P * input.r;

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let part_C1 = -(srs.X1P[N] - srs.X1P[0]) * r[2] - srs.X1P[0] * r[0];
    let part_C3 = -(srs.X1P[n] - srs.X1P[0]) * r[6];
    let mut C = vec![
      part_C1 + model.g1 * r[1] + A_x * model.r + (srs.Y1P * model.r * r[1]) + srs.X1P[0] * r[1] * beta,
      -srs.X1P[1] * r[4] + srs.X1P[0] * (r[1] - r[3]),
      part_C3 + input.g1 * r[5] + B_x * input.r + srs.X1P[0] * (r[5] * beta) + srs.Y1P * input.r * r[5],
      -srs.X1P[1] * r[7] + srs.X1P[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap()),
      -srs.X1P[0] * r[8] + srs.X1P[N - n] * r[5],
    ];
    proof.append(&mut C);
    let mut betas: Vec<Fr> = vec![beta];

    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![part_C1, part_C3, model.g1, A_x, input.g1, B_x];

      proof.append(&mut additional_g1_for_acc);
      betas.append(&mut vec![r[1], model.r, input.r, r[5]]);
    }

    return (proof, vec![setup.1[0].into(), f_x_2], betas);
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = model.first().unwrap().len;
    let n = inputs[0].first().unwrap().len;
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    let beta = Fr::rand(rng);
    println!("verify A_x: {:?}", A_x);
    println!("verify A_zero: {:?}", A_zero);
    println!("verify A_zero_div: {:?}", A_zero_div);
    println!("verify C2: {:?}", C2);

    // Check A(x) (A_i = m_i/(t_i+beta))
    checks.push(vec![
      (A_x, T_x_2),
      ((A_x * beta - m_x).into(), srs.X2A[0]),
      (-A_Q_x, (srs.X2A[N] - srs.X2A[0]).into()),
      (-C1, srs.Y2A),
    ]);

    // Check T_x_2 is the G2 equivalent of the model
    checks.push(vec![(model.first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -T_x_2)]);

    // Check A(x) - A(0) is divisible by x
    checks.push(vec![((A_x - A_zero).into(), srs.X2A[0]), (-A_zero_div, srs.X2A[1]), (-C2, srs.Y2A)]);

    // Check B(x) (B_i = 1/(f_i+beta))
    checks.push(vec![
      (B_x, f_x_2),
      ((B_x * beta - srs.X1A[0]).into(), srs.X2A[0]),
      (-B_Q_x, (srs.X2A[n] - srs.X2A[0]).into()),
      (-C3, srs.Y2A),
    ]);

    // Check f_x_2 is the G2 equivalent of the input
    checks.push(vec![(inputs[0].first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -f_x_2)]);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0: G1Affine = (A_zero * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    checks.push(vec![((B_x - B_0).into(), srs.X2A[0]), (-B_zero_div, srs.X2A[1]), (-C4, srs.Y2A)]);

    // Degree check B
    checks.push(vec![(B_x, srs.X2A[N - n]), (-B_DC, srs.X2A[0]), (-C5, srs.Y2A)]);
    checks
  }

  fn acc_init(
    &self,
    srs: &SRS,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), Vec::new());
    let g1_zero = G1Projective::zero();
    let g2_zero = G2Projective::zero();
    let fr_zero = Fr::zero();

    // Generate Fiat-Shamir challenge
    let mut bytes = Vec::new();
    proof.0[..proof.0.len() - 6].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    proof.2[..1].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);

    let [m_x, A_x, A_Q_x, _A_zero, _A_zero_div, B_x, B_Q_x, _B_zero_div, _B_DC, C1, _C2, C3, _C4, _C5, part_C1, part_C3, model_g1, A_x_1, input_g1, B_x_1] =
      proof.0[..]
    else {
      panic!("Wrong proof format")
    };

    println!("acc init A_x: {:?}", A_x);
    println!("acc init A_zero: {:?}", _A_zero);
    println!("acc init A_zero_div: {:?}", _A_zero_div);
    println!("acc init C2: {:?}", _C2);
    // Initialize accumulators with zero values
    acc_proof.0.extend(vec![g1_zero; 12 * 2]); // For error terms
    acc_proof.1.extend(vec![g2_zero; 4 * 2]); // For G2 elements
    acc_proof.2.extend(vec![fr_zero; 5]); // beta, A_r, model_r, input_r, B_r

    // mu
    acc_proof.2.push(Fr::one());

    acc_proof
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let [m_x, A_x, A_Q_x, _A_zero, _A_zero_div, B_x, B_Q_x, _B_zero_div, _B_DC, C1, _C2, C3, _C4, _C5, part_C1, part_C3, model_g1, A_x_1, input_g1, B_x_1] =
      proof.0[..]
    else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };
    let [beta, A_r, model_r, input_r, B_r] = proof.2[..] else {
      panic!("Wrong proof format")
    };

    // Extract values from acc_proof
    let acc_holder = acc_proof_to_cq_acc(acc_proof, true);
    let [acc_m_x, acc_A_x, acc_A_Q_x, _acc_A_zero, _acc_A_zero_div, acc_B_x, acc_B_Q_x, _acc_B_zero_div, _acc_B_DC, _acc_C1, _acc_C2, _acc_C3, _acc_C4, _acc_C5, acc_part_C1, acc_part_C3, acc_model_g1, acc_A_x_1, acc_input_g1, acc_B_x_1] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };

    let A_x_affine = A_x.into_affine();
    let acc_A_x_affine = acc_A_x.into_affine();
    let A_zero_affine = _A_zero.into_affine();
    let acc_A_zero_affine = _acc_A_zero.into_affine();
    let A_zero_div_affine = _A_zero_div.into_affine();
    let acc_A_zero_div_affine = _acc_A_zero_div.into_affine();
    let C2_affine = _C2.into_affine();
    let acc_C2_affine = _acc_C2.into_affine();
    println!("acc prove A_x: {:?}, acc_A_x: {:?}", A_x_affine, acc_A_x_affine);
    println!("acc prove A_zero: {:?}, acc_A_zero: {:?}", A_zero_affine, acc_A_zero_affine);
    println!(
      "acc prove A_zero_div: {:?}, acc_A_zero_div: {:?}",
      A_zero_div_affine, acc_A_zero_div_affine
    );
    println!("acc prove C2: {:?}, acc_C2: {:?}", C2_affine, acc_C2_affine);
    let [acc_T_x_2, acc_f_x_2] = acc_holder.acc_g2[..] else {
      panic!("Wrong proof format")
    };

    let acc_holder = acc_proof_to_cq_acc(acc_proof, true);
    let acc_mu = acc_holder.mu;
    let [acc_beta, acc_A_r, acc_model_r, acc_input_r, acc_B_r] = acc_holder.acc_fr[..] else {
      panic!("Wrong proof format")
    };

    let err_1 = (
      vec![
        A_x,
        acc_A_x,
        acc_A_x * beta,
        A_x * acc_beta,
        acc_A_Q_x + A_Q_x * acc_mu,
        m_x * acc_mu + acc_m_x,
        acc_part_C1
          + part_C1 * acc_mu
          + acc_A_x_1 * model_r
          + A_x_1 * acc_model_r
          + acc_model_g1 * A_r
          + model_g1 * acc_A_r
          + srs.X1P[0] * (beta * acc_A_r + acc_beta * A_r)
          + srs.Y1P * (acc_model_r * A_r + acc_A_r * model_r),
      ],
      vec![acc_T_x_2, T_x_2],
      vec![],
    );

    let err_3 = (
      vec![
        B_x,
        acc_B_x,
        acc_B_Q_x + B_Q_x * acc_mu,
        B_x * acc_mu + acc_B_x,
        acc_part_C3
          + part_C3 * acc_mu
          + acc_input_g1 * B_r
          + input_g1 * acc_B_r
          + acc_B_x_1 * input_r
          + B_x_1 * acc_input_r
          + srs.X1P[0] * (acc_input_r * beta + acc_beta * input_r)
          + srs.Y1P * (acc_input_r * B_r + acc_B_r * input_r),
      ],
      vec![acc_f_x_2, f_x_2],
      vec![-Fr::from(2) * acc_mu],
    );

    // Combine error terms
    let mut errs = vec![err_1, err_3];

    // Generate Fiat-Shamir challenge
    let mut bytes = Vec::new();
    acc_holder.acc_g1[..acc_holder.acc_g1.len() - 11].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_fr[..1].serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..proof.0.len() - 6].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    proof.2[..1].serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    // Create new accumulator
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };
    new_acc_holder.acc_g1 = proof.0.iter().zip(acc_proof.0.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.acc_g2 = proof.1.iter().zip(acc_proof.1.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.acc_fr = proof.2.iter().zip(acc_proof.2.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.mu = acc_mu + acc_gamma;
    new_acc_holder.errs = errs.clone();
    new_acc_holder.acc_errs = acc_holder.acc_errs;

    for i in 0..errs.len() {
      errs[i].0 = errs[i].0.iter().map(|x| (*x * acc_gamma).into()).collect();
      errs[i].2 = errs[i].2.iter().map(|x| (*x * acc_gamma).into()).collect();
    }

    let err1_g1_len = new_acc_holder.acc_errs[0].0.len();
    let A_Q_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 3].clone() + errs[0].0[4];
    let m_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 2].clone() + errs[0].0[5];
    let c1_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 1].clone() + errs[0].0[6];
    let mut errs_0_g1 = errs[0].0[..4].to_vec();
    let mut errs_0_g2 = errs[0].1[..2].to_vec();

    new_acc_holder.acc_errs[0].0 = new_acc_holder.acc_errs[0].0[..err1_g1_len - 3].to_vec();
    new_acc_holder.acc_errs[0].0.append(&mut errs_0_g1);
    new_acc_holder.acc_errs[0].0.push(A_Q_term_g1);
    new_acc_holder.acc_errs[0].0.push(m_term_g1);
    new_acc_holder.acc_errs[0].0.push(c1_term_g1);
    new_acc_holder.acc_errs[0].1.append(&mut errs_0_g2);

    let err3_g1_len = new_acc_holder.acc_errs[1].0.len();
    let B_Q_term_g1 = new_acc_holder.acc_errs[1].0[err3_g1_len - 3].clone() + errs[1].0[2];
    let B_term_g1 = new_acc_holder.acc_errs[1].0[err3_g1_len - 2].clone() + errs[1].0[3];
    let c3_term_g1 = new_acc_holder.acc_errs[1].0[err3_g1_len - 1].clone() + errs[1].0[4];
    let mut errs_1_g1 = errs[1].0[..2].to_vec();
    let mut errs_1_g2 = errs[1].1[..2].to_vec();

    new_acc_holder.acc_errs[1].0 = new_acc_holder.acc_errs[1].0[..err3_g1_len - 3].to_vec();
    new_acc_holder.acc_errs[1].0.append(&mut errs_1_g1);
    new_acc_holder.acc_errs[1].0.push(B_Q_term_g1);
    new_acc_holder.acc_errs[1].0.push(B_term_g1);
    new_acc_holder.acc_errs[1].0.push(c3_term_g1);
    new_acc_holder.acc_errs[1].1.append(&mut errs_1_g2);
    new_acc_holder.acc_errs[1].2[0] = errs[1].2[0];

    acc_to_acc_proof(new_acc_holder)
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    // acc_part_C1
    //       + part_C1 * acc_mu
    //       + acc_M_1 * cqlin_mask_A
    //       + M_x_1 * acc_mask_A
    //       + acc_A_1 * cqlin_mask_M
    //       + A_x_1 * acc_mask_M
    //       + srs.Y1P * (cqlin_mask_A * acc_mask_M + cqlin_mask_M * acc_mask_A),
    let mut acc_holder = acc_proof_to_cq_acc(acc_proof, true);
    let [_acc_m_x, acc_A_x, _acc_A_Q_x, _acc_A_zero, _acc_A_zero_div, _acc_B_x, _acc_B_Q_x, _acc_B_zero_div, _acc_B_DC, _acc_C1, _acc_C2, _acc_C3, _acc_C4, _acc_C5, acc_part_C1, acc_part_C3, acc_model_g1, acc_A_x_1, acc_input_g1, acc_B_x_1] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };
    let [acc_beta, acc_A_r, acc_model_r, acc_input_r, acc_B_r] = acc_holder.acc_fr[..] else {
      panic!("Wrong proof format")
    };
    let acc_A_x_affine = acc_A_x.into_affine();
    let acc_A_zero_affine = _acc_A_zero.into_affine();
    let acc_A_zero_div_affine = _acc_A_zero_div.into_affine();
    let acc_C2_affine = _acc_C2.into_affine();
    println!("clean acc_A_x: {:?}", acc_A_x_affine);
    println!("clean acc_A_zero: {:?}", acc_A_zero_affine);
    println!("clean acc_A_zero_div: {:?}", acc_A_zero_div_affine);
    println!("clean acc_C2: {:?}", acc_C2_affine);
    acc_holder.acc_g1[9] = acc_part_C1 * acc_holder.mu
      + acc_model_g1 * acc_A_r
      + acc_A_x_1 * acc_model_r
      + srs.Y1P * acc_A_r * acc_model_r
      + srs.X1P[0] * (acc_beta * acc_A_r);
    acc_holder.acc_g1[11] = acc_part_C3 * acc_holder.mu
      + acc_input_g1 * acc_B_r
      + srs.Y1P * acc_B_r * acc_input_r
      + acc_B_x_1 * acc_input_r
      + srs.X1P[0] * (acc_input_r * acc_B_r);
    // correct the blinding factor C1
    // acc_holder.acc_g1[9] = acc_holder.acc_g1[acc_holder.acc_g1.len() - 3] * acc_holder.mu
    //   + acc_holder.acc_g1[acc_holder.acc_g1.len() - 2] * acc_holder.acc_fr[1]
    //   + srs.Y1P * acc_holder.acc_fr[1] * acc_holder.acc_fr[log_n + 2]
    //   + acc_holder.acc_g1[acc_holder.acc_g1.len() - 1] * acc_holder.acc_fr[log_n + 2];
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_holder.acc_g1[..acc_holder.acc_g1.len() - 6].to_vec();
    acc_holder.acc_fr = acc_holder.acc_fr[..1].to_vec();
    let acc_proof = acc_to_acc_proof(acc_holder);

    // Remove blinding factors from proofs
    let clean_proof = (
      proof.0[..proof.0.len() - 6].iter().map(|x| (*x).into()).collect(),
      proof.1.iter().map(|x| (*x).into()).collect(),
      proof.2[..1].iter().map(|x| (*x).into()).collect(),
    );

    let clean_acc = (
      acc_proof.0.iter().map(|x| (*x).into()).collect(),
      acc_proof.1.iter().map(|x| (*x).into()).collect(),
      acc_proof.2.clone(),
    );

    (clean_proof, clean_acc)
  }

  fn acc_verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    println!("acc verify A_x: {:?}", A_x);
    println!("acc verify A_zero: {:?}", A_zero);
    println!("acc verify A_zero_div: {:?}", A_zero_div);
    println!("acc verify C2: {:?}", C2);

    // Verify that acc_proof is a valid accumulation of prev_acc_proof and proof
    let prev_acc_holder = acc_proof_to_cq_acc(prev_acc_proof, false);
    let acc_holder = acc_proof_to_cq_acc(acc_proof, false);

    let beta = Fr::rand(rng);
    let mut result = beta == proof.2[0];
    println!("result: {:?}", result);

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      let mut bytes = Vec::new();
      proof.0.serialize_uncompressed(&mut bytes).unwrap();
      proof.1.serialize_uncompressed(&mut bytes).unwrap();
      proof.2.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_holder.acc_g1[..prev_acc_holder.acc_g1.len() - 5].serialize_uncompressed(&mut bytes).unwrap();
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

    let cq_proof_g1 = [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5];
    let cq_proof_g2 = [T_x_2, f_x_2];
    cq_proof_g1.iter().zip(prev_acc_holder.acc_g1.iter()).enumerate().for_each(|(i, (x, y))| {
      if i >= 9 {
        return;
      }
      let z = *y + *x * acc_gamma;
      let z: G1Affine = z.into();
      result &= z == acc_holder.acc_g1[i];
      println!("g1 result: {:?}", result);
    });
    cq_proof_g2.iter().zip(prev_acc_holder.acc_g2.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *y + *x * acc_gamma;
      let z: G2Affine = z.into();
      result &= z == acc_holder.acc_g2[i];
      println!("g2 result: {:?}", result);
    });
    proof.2.iter().zip(prev_acc_holder.acc_fr.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *y + *x * acc_gamma;
      result &= z == acc_holder.acc_fr[i];
      println!("fr result: {:?}", result);
    });

    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let N = srs.X2P.len() - 1;
    let n = self.n;

    let acc_holder = acc_proof_to_cq_acc(acc_proof, false);

    let [acc_m_x, acc_A_x, acc_A_Q_x, acc_A_zero, acc_A_zero_div, acc_B_x, acc_B_Q_x, acc_B_zero_div, acc_B_DC, acc_C1, acc_C2, acc_C3, acc_C4, acc_C5] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };

    println!("decide acc_A_x: {:?}", acc_A_x);
    println!("decide acc_A_zero: {:?}", acc_A_zero);
    println!("decide acc_A_zero_div: {:?}", acc_A_zero_div);
    println!("decide acc_C2: {:?}", acc_C2);

    let acc_mu = acc_holder.mu;
    let acc_beta = acc_holder.acc_fr[0];
    let err_1 = &acc_holder.acc_errs[0];
    let err_3 = &acc_holder.acc_errs[1];

    let mut err1: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      err1.push((-err_1.0[i], err_1.1[i]));
    }
    err1.push((-err_1.0[err_1.1.len()], srs.X2A[0]));
    err1.push((-err_1.0[err_1.1.len() + 1], srs.X2A[0]));
    err1.push((err_1.0[err_1.1.len() + 2], (srs.X2A[N] - srs.X2A[0]).into()));
    err1.push((err_1.0[err_1.1.len() + 3], srs.X2A[0]));
    err1.push((err_1.0[err_1.1.len() + 4], srs.Y2A));
    let mut acc_1: PairingCheck = vec![
      (acc_A_x, acc_holder.acc_g2[0]),
      ((-acc_m_x * acc_mu + acc_A_x * acc_beta).into(), srs.X2A[0]),
      ((-acc_A_Q_x * acc_mu).into(), (srs.X2A[N] - srs.X2A[0]).into()),
      (-acc_C1, srs.Y2A),
    ];
    acc_1.extend(err1);

    let acc_2: PairingCheck = vec![
      ((acc_A_x - acc_A_zero).into(), srs.X2A[0]),
      (-acc_A_zero_div, srs.X2A[1]),
      (-acc_C2, srs.Y2A),
    ];

    //  Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let acc_B_0: G1Affine = (acc_A_zero * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    let mut err3: PairingCheck = vec![];
    for i in 0..err_3.1.len() {
      err3.push((-err_3.0[i], err_3.1[i]));
    }
    err3.push((err_3.0[err_3.1.len()], (srs.X2A[n] - srs.X2A[0]).into()));
    err3.push((-err_3.0[err_3.1.len() + 1], srs.X2A[0]));
    err3.push(((srs.X1A[0] * err_3.2[0]).into(), srs.X2A[0]));
    let mut acc_3: PairingCheck = vec![
      (acc_B_x, acc_holder.acc_g2[1]),
      ((acc_B_x * acc_mu - srs.X1P[0] * acc_mu * acc_mu).into(), srs.X2A[0]),
      (-acc_B_Q_x, (srs.X2A[n] - srs.X2A[0]).into()),
      (-acc_C3, srs.Y2A),
    ];
    acc_3.extend(err3);

    // Check B(x) - B(0) is divisible by x
    let acc_4 = vec![
      ((acc_B_x - acc_B_0).into(), srs.X2A[0]),
      (-acc_B_zero_div, srs.X2A[1]),
      (-acc_C4, srs.Y2A),
    ];

    // Degree check B
    let acc_5 = vec![(acc_B_x, srs.X2A[N - n]), (-acc_B_DC, srs.X2A[0]), (-acc_C5, srs.Y2A)];

    // let checks = vec![acc_1, acc_4];
    let checks = vec![acc_4];

    checks
  }

  fn acc_clean_errs(&self, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) {
    let mut acc_holder = acc_proof_to_cq_acc(acc_proof, false);
    acc_holder.errs = vec![];
    acc_to_acc_proof(acc_holder)
  }
}
