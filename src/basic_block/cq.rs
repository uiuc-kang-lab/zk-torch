#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_proof_to_acc, acc_to_acc_proof, get_cq_N, AccHolder, AccProofLayout};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::AffineRepr;
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

pub fn cq_acc_clean(
  bb: &dyn AccProofLayout,
  srs: &SRS,
  proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  acc_proof: (
    &Vec<G1Projective>,
    &Vec<G2Projective>,
    &Vec<Fr>,
    &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
  ),
) -> (
  (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>),
  (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>),
) {
  let mut acc_holder = acc_proof_to_acc(bb, acc_proof, true);
  let [acc_part_C1, acc_part_C3, acc_model_g1, acc_input_g1, acc_A_x_1, acc_B_x_1] = acc_holder.acc_g1[acc_holder.acc_g1.len() - 6..] else {
    panic!("Wrong proof format")
  };
  let [acc_beta, acc_model_r, acc_input_r, acc_A_r, acc_B_r] = acc_holder.acc_fr[..] else {
    panic!("Wrong proof format")
  };
  acc_holder.acc_g1[9] = acc_part_C1 * acc_holder.mu
    + acc_model_g1 * acc_A_r
    + acc_A_x_1 * acc_model_r
    + srs.Y1P * acc_A_r * acc_model_r
    + srs.X1P[0] * (acc_beta * acc_A_r);
  acc_holder.acc_g1[11] = acc_part_C3 * acc_holder.mu
    + acc_input_g1 * acc_B_r
    + acc_B_x_1 * acc_input_r
    + srs.Y1P * acc_B_r * acc_input_r
    + srs.X1P[0] * (acc_beta * acc_B_r);
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
    acc_proof.3.iter().map(|x| *x).collect(),
  );

  (clean_proof, clean_acc)
}

pub fn cq_acc_decide(
  bb: &dyn AccProofLayout,
  srs: &SRS,
  acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  N: usize,
  n: usize,
) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
  let acc_holder = acc_proof_to_acc(bb, acc_proof, false);

  let [acc_m_x, acc_A_x, acc_A_Q_x, acc_A_zero, acc_A_zero_div, acc_B_x, acc_B_Q_x, acc_B_zero_div, acc_B_DC, acc_C1, acc_C2, acc_C3, acc_C4, acc_C5, acc_model, acc_input] =
    acc_holder.acc_g1[..]
  else {
    panic!("Wrong proof format")
  };

  let [acc_T_x_2, acc_f_x_2] = acc_holder.acc_g2[..] else {
    panic!("Wrong proof format")
  };

  let acc_mu = acc_holder.mu;
  let acc_beta = acc_holder.acc_fr[0];
  let err_1 = &acc_holder.acc_errs[0];
  let err_3 = &acc_holder.acc_errs[1];

  let mut err1: PairingCheck = vec![];
  err1.push((err_1.0[0], (srs.X2A[N] - srs.X2A[0]).into()));
  err1.push((-err_1.0[1], srs.X2A[0]));
  err1.push((err_1.0[2], srs.Y2A));
  let mut acc_1: PairingCheck = vec![
    (acc_A_x, acc_T_x_2),
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
  err3.push((err_3.0[0], (srs.X2A[n] - srs.X2A[0]).into()));
  err3.push((-err_3.0[1], srs.X2A[0]));
  err3.push((err_3.0[2], srs.Y2A));
  let mut acc_3: PairingCheck = vec![
    (acc_B_x, acc_f_x_2),
    ((acc_B_x * acc_beta - srs.X1P[0] * acc_mu * acc_mu).into(), srs.X2A[0]),
    ((-acc_B_Q_x * acc_mu).into(), (srs.X2A[n] - srs.X2A[0]).into()),
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

  // Check T_x_2 is the G2 equivalent of the model
  let acc_6 = vec![(acc_model, srs.X2A[0]), (srs.X1A[0], -acc_T_x_2)];

  // Check f_x_2 is the G2 equivalent of the input
  let acc_7 = vec![(acc_input, srs.X2A[0]), (srs.X1A[0], -acc_f_x_2)];

  let checks = vec![
    (acc_1, err_1.3[0]),
    (acc_2, PairingOutput::<Bn<ark_bn254::Config>>::zero()),
    (acc_3, err_3.3[0]),
    (acc_4, PairingOutput::<Bn<ark_bn254::Config>>::zero()),
    (acc_5, PairingOutput::<Bn<ark_bn254::Config>>::zero()),
    (acc_6, PairingOutput::<Bn<ark_bn254::Config>>::zero()),
    (acc_7, PairingOutput::<Bn<ark_bn254::Config>>::zero()),
  ];
  checks
}

pub fn cq_acc_finalize(
  bb: &dyn AccProofLayout,
  srs: &SRS,
  acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  N: usize,
  n: usize,
) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
  let mut acc_holder = acc_proof_to_acc(bb, acc_proof, false);

  let tmp_1 = &acc_holder.acc_errs[0];
  let tmp_3 = &acc_holder.acc_errs[1];

  let mut err1: PairingCheck = vec![];
  err1.push((tmp_1.0[0], (srs.X2A[N] - srs.X2A[0]).into()));
  err1.push((-tmp_1.0[1], srs.X2A[0]));
  err1.push((tmp_1.0[2], srs.Y2A));
  let pairing_1: Vec<_> = err1.iter().map(|x| x).collect();
  let pairing_1: (Vec<_>, Vec<_>) = (pairing_1.iter().map(|x| x.0).collect(), pairing_1.iter().map(|x| x.1).collect());
  let err1 = Bn254::multi_pairing(pairing_1.0.iter(), pairing_1.1.iter());

  let mut err3: PairingCheck = vec![];
  err3.push((tmp_3.0[0], (srs.X2A[n] - srs.X2A[0]).into()));
  err3.push((-tmp_3.0[1], srs.X2A[0]));
  err3.push((tmp_3.0[2], srs.Y2A));
  let pairing_3: Vec<_> = err3.iter().map(|x| x).collect();
  let pairing_3: (Vec<_>, Vec<_>) = (pairing_3.iter().map(|x| x.0).collect(), pairing_3.iter().map(|x| x.1).collect());
  let err3 = Bn254::multi_pairing(pairing_3.0.iter(), pairing_3.1.iter());

  acc_holder.errs = vec![];
  acc_holder.acc_errs = vec![];
  let acc_proof = acc_to_acc_proof(acc_holder);
  (acc_proof.0, acc_proof.1, acc_proof.2, vec![err1, err3])
}

pub struct CQLayoutHelper;

impl CQLayoutHelper {
  pub fn acc_g1_num(is_prover: bool) -> usize {
    if is_prover {
      22
    } else {
      16
    }
  }
  pub fn acc_g2_num() -> usize {
    2
  }
  pub fn acc_fr_num(is_prover: bool) -> usize {
    if is_prover {
      5
    } else {
      1
    }
  }
  pub fn err_g1_nums() -> Vec<usize> {
    vec![3, 3]
  }
  pub fn err_g2_nums() -> Vec<usize> {
    vec![0, 0]
  }
  pub fn err_fr_nums() -> Vec<usize> {
    vec![0, 0]
  }
  pub fn err_gt_nums() -> Vec<usize> {
    vec![1, 1]
  }
  pub fn prover_proof_to_acc(proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    let group_errs = vec![
      (
        vec![G1Projective::zero(); 3],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      ),
      (
        vec![G1Projective::zero(); 3],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      ),
    ];
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: proof.2.clone(),
      mu: Fr::one(),
      errs: group_errs.clone(),
      acc_errs: group_errs,
    }
  }
  pub fn verifier_proof_to_acc(proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    let group_errs = vec![
      (
        vec![G1Affine::zero(); 3],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      ),
      (
        vec![G1Affine::zero(); 3],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      ),
    ];
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: proof.2.clone(),
      mu: Fr::one(),
      errs: group_errs.clone(),
      acc_errs: group_errs,
    }
  }
  pub fn mira_prove(
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    let [m_x, A_x, A_Q_x, _A_zero, _A_zero_div, B_x, B_Q_x, _B_zero_div, _B_DC, _C1, _C2, _C3, _C4, _C5, _model_g1_blinded, _input_g1_blinded, part_C1, part_C3, model_g1, input_g1, A_x_1, B_x_1] =
      acc_2.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = acc_2.acc_g2[..] else {
      panic!("Wrong proof format")
    };
    let [beta, model_r, input_r, A_r, B_r] = acc_2.acc_fr[..] else {
      panic!("Wrong proof format")
    };

    let [acc_m_x, acc_A_x, acc_A_Q_x, _acc_A_zero, _acc_A_zero_div, acc_B_x, acc_B_Q_x, _acc_B_zero_div, _acc_B_DC, _acc_C1, _acc_C2, _acc_C3, _acc_C4, _acc_C5, _acc_model_g1_blinded, _acc_input_g1_blinded, acc_part_C1, acc_part_C3, acc_model_g1, acc_input_g1, acc_A_x_1, acc_B_x_1] =
      acc_1.acc_g1[..]
    else {
      panic!("Wrong proof format")
    };
    let [acc_T_x_2, acc_f_x_2] = acc_1.acc_g2[..] else {
      panic!("Wrong proof format")
    };
    let acc_mu = acc_1.mu;
    let [acc_beta, acc_model_r, acc_input_r, acc_A_r, acc_B_r] = acc_1.acc_fr[..] else {
      panic!("Wrong proof format")
    };

    let err_1 = (
      vec![
        acc_A_Q_x * acc_2.mu + A_Q_x * acc_mu,
        acc_A_x * beta + A_x * acc_beta - m_x * acc_mu - acc_m_x * acc_2.mu,
        acc_part_C1 * acc_2.mu
          + part_C1 * acc_mu
          + acc_A_x_1 * model_r
          + A_x_1 * acc_model_r
          + acc_model_g1 * A_r
          + model_g1 * acc_A_r
          + srs.X1P[0] * (beta * acc_A_r + acc_beta * A_r)
          + srs.Y1P * (acc_model_r * A_r + acc_A_r * model_r),
      ],
      vec![],
      vec![],
      vec![Bn254::multi_pairing(vec![A_x, acc_A_x], vec![acc_T_x_2, T_x_2])],
    );

    let err_3 = (
      vec![
        acc_B_Q_x * acc_2.mu + B_Q_x * acc_mu,
        acc_B_x * beta + B_x * acc_beta - srs.X1P[0] * Fr::from(2) * acc_mu * acc_2.mu,
        acc_part_C3 * acc_2.mu
          + part_C3 * acc_mu
          + acc_input_g1 * B_r
          + input_g1 * acc_B_r
          + acc_B_x_1 * input_r
          + B_x_1 * acc_input_r
          + srs.X1P[0] * (acc_B_r * beta + acc_beta * B_r)
          + srs.Y1P * (acc_input_r * B_r + acc_B_r * input_r),
      ],
      vec![],
      vec![],
      vec![Bn254::multi_pairing(vec![B_x, acc_B_x], vec![acc_f_x_2, f_x_2])],
    );

    // Combine error terms
    let errs = vec![err_1, err_3];

    // Generate Fiat-Shamir challenge
    let mut bytes = Vec::new();
    acc_1.acc_g1[..acc_1.acc_g1.len() - 13].serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_fr[..1].serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g1[..acc_2.acc_g1.len() - 13].serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_fr[..1].serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f, gt)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
      gt.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);
    let acc_gamma_sq = acc_gamma * acc_gamma;

    // Create new accumulator
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };
    new_acc_holder.acc_g1 = acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.acc_g2 = acc_2.acc_g2.iter().zip(acc_1.acc_g2.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.acc_fr = acc_2.acc_fr.iter().zip(acc_1.acc_fr.iter()).map(|(x, y)| *x * acc_gamma + *y).collect();
    new_acc_holder.mu = acc_mu + acc_gamma * acc_2.mu;
    new_acc_holder.errs = errs;
    new_acc_holder.acc_errs = acc_1.acc_errs;

    let A_Q_term_g1 = new_acc_holder.acc_errs[0].0[0].clone() + new_acc_holder.errs[0].0[0] * acc_gamma + acc_2.acc_errs[0].0[0] * acc_gamma_sq;
    let m_term_g1 = new_acc_holder.acc_errs[0].0[1].clone() + new_acc_holder.errs[0].0[1] * acc_gamma + acc_2.acc_errs[0].0[1] * acc_gamma_sq;
    let c1_term_g1 = new_acc_holder.acc_errs[0].0[2].clone() + new_acc_holder.errs[0].0[2] * acc_gamma + acc_2.acc_errs[0].0[2] * acc_gamma_sq;
    let term_gt = new_acc_holder.acc_errs[0].3[0] + new_acc_holder.errs[0].3[0] * acc_gamma + acc_2.acc_errs[0].3[0] * acc_gamma_sq;

    let B_Q_term_g1 = new_acc_holder.acc_errs[1].0[0].clone() + new_acc_holder.errs[1].0[0] * acc_gamma + acc_2.acc_errs[1].0[0] * acc_gamma_sq;
    let B_term_g1 = new_acc_holder.acc_errs[1].0[1].clone() + new_acc_holder.errs[1].0[1] * acc_gamma + acc_2.acc_errs[1].0[1] * acc_gamma_sq;
    let c3_term_g1 = new_acc_holder.acc_errs[1].0[2].clone() + new_acc_holder.errs[1].0[2] * acc_gamma + acc_2.acc_errs[1].0[2] * acc_gamma_sq;
    let term_gt2 = new_acc_holder.acc_errs[1].3[0] + new_acc_holder.errs[1].3[0] * acc_gamma + acc_2.acc_errs[1].3[0] * acc_gamma_sq;

    new_acc_holder.acc_errs = vec![
      (vec![A_Q_term_g1, m_term_g1, c1_term_g1], vec![], vec![], vec![term_gt]),
      (vec![B_Q_term_g1, B_term_g1, c3_term_g1], vec![], vec![], vec![term_gt2]),
    ];

    new_acc_holder
  }
  pub fn mira_verify(
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    let mut result = true;
    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_1.acc_g1[..acc_1.acc_g1.len() - 7].serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_fr.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g1[..acc_2.acc_g1.len() - 7].serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_fr.serialize_uncompressed(&mut bytes).unwrap();
    new_acc.errs.iter().for_each(|(g1, g2, f, gt)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
      gt.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);
    let acc_gamma_sq = acc_gamma * acc_gamma;

    acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).enumerate().for_each(|(i, (x, y))| {
      if i >= 9 {
        return;
      }
      let z = *y + *x * acc_gamma;
      let z: G1Affine = z.into();
      result &= z == new_acc.acc_g1[i];
    });
    acc_2.acc_g2.iter().zip(acc_1.acc_g2.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *y + *x * acc_gamma;
      let z: G2Affine = z.into();
      result &= z == new_acc.acc_g2[i];
    });
    acc_2.acc_fr.iter().zip(acc_1.acc_fr.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *y + *x * acc_gamma;
      result &= z == new_acc.acc_fr[i];
    });

    // Check RLC for errors
    for i in 0..2 {
      new_acc.errs[i].0.iter().zip(acc_1.acc_errs[i].0.iter()).enumerate().for_each(|(j, (x, y))| {
        let z = *y + *x * acc_gamma + acc_2.acc_errs[i].0[j] * acc_gamma_sq;
        result &= z == new_acc.acc_errs[i].0[j];
      });
      result &= acc_1.acc_errs[i].3[0] + new_acc.errs[i].3[0] * acc_gamma + acc_2.acc_errs[i].3[0] * acc_gamma_sq == new_acc.acc_errs[i].3[0];
    }

    Some(result)
  }
}

impl AccProofLayout for CQBasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    CQLayoutHelper::acc_g1_num(is_prover)
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    CQLayoutHelper::acc_g2_num()
  }
  fn acc_fr_num(&self, is_prover: bool) -> usize {
    CQLayoutHelper::acc_fr_num(is_prover)
  }
  fn err_g1_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_g1_nums()
  }
  fn err_g2_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_g2_nums()
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_fr_nums()
  }
  fn err_gt_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_gt_nums()
  }
  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    CQLayoutHelper::prover_proof_to_acc(proof)
  }
  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    CQLayoutHelper::verifier_proof_to_acc(proof)
  }
  fn mira_prove(
    &self,
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    CQLayoutHelper::mira_prove(srs, acc_1, acc_2, rng)
  }
  fn mira_verify(
    &self,
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    CQLayoutHelper::mira_verify(acc_1, acc_2, new_acc, rng)
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
    let mut fr: Vec<Fr> = vec![beta];

    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![
        model.g1 + srs.Y1P * model.r,
        input.g1 + srs.Y1P * input.r,
        part_C1,
        part_C3,
        model.g1,
        input.g1,
        A_x,
        B_x,
      ];

      proof.append(&mut additional_g1_for_acc);
      fr.append(&mut vec![model.r, input.r, r[1], r[5]]);
    }

    return (proof, vec![setup.1[0].into(), f_x_2], fr);
  }

  #[cfg(not(feature = "fold"))]
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
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5] = proof.0[..14] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    let beta = Fr::rand(rng);

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

  #[cfg(feature = "fold")]
  fn verify(
    &self,
    _srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let model = model.first().unwrap().g1;
    let input = inputs[0].first().unwrap().g1;

    let beta = Fr::rand(rng);
    let mut result = beta == proof.2[0];
    result &= model == proof.0[proof.0.len() - 2];
    result &= input == proof.0[proof.0.len() - 1];
    //assert!(result, "acc_proof for cq is not valid");
    vec![]
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let proof = self.prover_proof_to_acc(proof);
    if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
      return acc_to_acc_proof(proof);
    }
    let acc_proof = acc_proof_to_acc(self, acc_proof, true);
    acc_to_acc_proof(self.mira_prove(srs, acc_proof, proof, rng))
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
  ) -> (
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>),
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) {
    cq_acc_clean(self, srs, proof, acc_proof)
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);

    let mut result = true;

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      return Some(result);
    }
    let proof = self.verifier_proof_to_acc(proof);
    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);
    result &= self.mira_verify(prev_acc_holder, proof, acc_holder, rng).unwrap();
    Some(result)
  }

  fn acc_decide(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let N = get_cq_N(&self.setup);
    let n = self.n;
    cq_acc_decide(self, srs, acc_proof, N, n)
  }

  fn acc_finalize(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let N = get_cq_N(&self.setup);
    let n = self.n;
    cq_acc_finalize(self, srs, acc_proof, N, n)
  }
}
