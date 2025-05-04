#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
use super::{
  AccProofAffine, AccProofAffineRef, AccProofProj, AccProofProjRef, BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS,
};
use crate::util::{self, acc_proof_to_holder, calc_pow, holder_to_acc_proof, AccHolder, AccProofLayout};
use crate::{define_acc_err_terms, define_acc_terms};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::AffineRepr;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use ndarray::{arr1, arr2, ArrayD, Ix1, Ix2, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::ParallelIterator;

define_acc_terms!(
  MatMulG1Terms,
  [
    Left_x,
    Left_Q_x,
    Left_zero,
    Left_zero_div,
    Right_x,
    Right_Q_x,
    Right_zero_div,
    Corr1,
    Corr2,
    Corr3,
    Corr4,
    Flat_A,
    Flat_B,
    Flat_C
  ],
  [Part_corr1, Flat_A_no_blind, Flat_B_no_blind]
);
define_acc_terms!(MatMulG2Terms, [Flat_B_g2, Beta_pow_g2], []);
define_acc_terms!(MatMulFrTerms, [], [Flat_A_r, Flat_B_r]);
define_acc_err_terms!(MatMulErrG1Terms, [Err_Q, Err_l, Err_C]);
define_acc_err_terms!(MatMulErrG2Terms, []);
define_acc_err_terms!(MatMulErrFrTerms, []);
define_acc_err_terms!(MatMulErrGtTerms, [Err_Gt]);

impl AccProofLayout for MatMulBasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    if is_prover {
      MatMulG1Terms::<G1Projective>::COUNT
    } else {
      MatMulG1Terms::<G1Projective>::PUBLIC_COUNT
    }
  }

  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    MatMulG2Terms::<G2Projective>::COUNT
  }

  fn acc_fr_num(&self, is_prover: bool) -> usize {
    if is_prover {
      MatMulFrTerms::<Fr>::COUNT
    } else {
      MatMulFrTerms::<Fr>::PUBLIC_COUNT
    }
  }

  fn err_g1_nums(&self) -> Vec<usize> {
    MatMulErrG1Terms::COUNTS.to_vec()
  }

  fn err_g2_nums(&self) -> Vec<usize> {
    MatMulErrG2Terms::COUNTS.to_vec()
  }

  fn err_fr_nums(&self) -> Vec<usize> {
    MatMulErrFrTerms::COUNTS.to_vec()
  }

  fn err_gt_nums(&self) -> Vec<usize> {
    MatMulErrGtTerms::COUNTS.to_vec()
  }

  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: proof.2.clone(),
      mu: Fr::one(),
      errs: vec![(
        vec![G1Projective::zero(); MatMulErrG1Terms::COUNTS[0]],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      )],
      acc_errs: vec![(
        vec![G1Projective::zero(); MatMulErrG1Terms::COUNTS[0]],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      )],
    }
  }

  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: proof.2.clone(),
      mu: Fr::one(),
      errs: vec![(
        vec![G1Affine::zero(); MatMulErrG1Terms::COUNTS[0]],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      )],
      acc_errs: vec![(
        vec![G1Affine::zero(); MatMulErrG1Terms::COUNTS[0]],
        vec![],
        vec![],
        vec![PairingOutput::<Bn<ark_bn254::Config>>::zero()],
      )],
    }
  }

  fn mira_prove(
    &self,
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    let acc_1_g1 = MatMulG1Terms::<G1Projective>::from_vec(&acc_1.acc_g1);
    let acc_1_g2 = MatMulG2Terms::<G2Projective>::from_vec(&acc_1.acc_g2);
    let acc_1_fr = MatMulFrTerms::<Fr>::from_vec(&acc_1.acc_fr);

    let acc_2_g1 = MatMulG1Terms::<G1Projective>::from_vec(&acc_2.acc_g1);
    let acc_2_g2 = MatMulG2Terms::<G2Projective>::from_vec(&acc_2.acc_g2);
    let acc_2_fr = MatMulFrTerms::<Fr>::from_vec(&acc_2.acc_fr);

    // Compute the error
    let err: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) = (
      vec![
        acc_1_g1.Left_Q_x * acc_2.mu + acc_2_g1.Left_Q_x * acc_1.mu,
        acc_1_g1.Left_x * acc_2.mu + acc_2_g1.Left_x * acc_1.mu,
        acc_1_g1.Part_corr1.unwrap() * acc_2.mu
          + acc_2_g1.Part_corr1.unwrap() * acc_1.mu
          + acc_1_g1.Flat_A_no_blind.unwrap() * acc_2_fr.Flat_B_r.unwrap()
          + acc_2_g1.Flat_A_no_blind.unwrap() * acc_1_fr.Flat_B_r.unwrap()
          + acc_1_g1.Flat_B_no_blind.unwrap() * acc_2_fr.Flat_A_r.unwrap()
          + acc_2_g1.Flat_B_no_blind.unwrap() * acc_1_fr.Flat_A_r.unwrap()
          + srs.Y1P * (acc_2_fr.Flat_A_r.unwrap() * acc_1_fr.Flat_B_r.unwrap() + acc_1_fr.Flat_A_r.unwrap() * acc_2_fr.Flat_B_r.unwrap()),
      ],
      vec![],
      vec![],
      vec![Bn254::multi_pairing(
        vec![acc_2_g1.Flat_A, acc_1_g1.Flat_A],
        vec![acc_1_g2.Flat_B_g2, acc_2_g2.Flat_B_g2],
      )],
    );
    let errs = vec![err];

    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };

    // Fiat-Shamir
    let mut bytes = Vec::new();
    let acc_1_g1_fiat_shamir = vec![
      acc_1_g1.Left_x,
      acc_1_g1.Left_Q_x,
      acc_1_g1.Left_zero,
      acc_1_g1.Left_zero_div,
      acc_1_g1.Right_x,
      acc_1_g1.Right_Q_x,
      acc_1_g1.Right_zero_div,
      acc_1_g1.Flat_A,
      acc_1_g1.Flat_B,
      acc_1_g1.Flat_C,
    ];
    acc_1_g1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_g2.serialize_uncompressed(&mut bytes).unwrap();

    let acc_2_g1_fiat_shamir = vec![
      acc_2_g1.Left_x,
      acc_2_g1.Left_Q_x,
      acc_2_g1.Left_zero,
      acc_2_g1.Left_zero_div,
      acc_2_g1.Right_x,
      acc_2_g1.Right_Q_x,
      acc_2_g1.Right_zero_div,
      acc_2_g1.Flat_A,
      acc_2_g1.Flat_B,
      acc_2_g1.Flat_C,
    ];
    acc_2_g1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f, gt)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
      gt.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);
    let acc_gamma_sq = acc_gamma * acc_gamma;

    new_acc_holder.acc_g1 = acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.acc_g2 = vec![acc_2_g2.Flat_B_g2 * acc_gamma + acc_1_g2.Flat_B_g2, acc_1_g2.Beta_pow_g2];
    new_acc_holder.acc_fr = acc_2.acc_fr.iter().zip(acc_1.acc_fr.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.mu = acc_1.mu + acc_gamma * acc_2.mu;
    new_acc_holder.errs = errs;

    // Append error terms
    let (q_group, q_idx) = MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_Q);
    let (l_group, l_idx) = MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_l);
    let (c_group, c_idx) = MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_C);
    let (gt_group, gt_idx) = MatMulErrGtTerms::idx(MatMulErrGtTerms::Err_Gt);
    let q_term_g1 =
      acc_1.acc_errs[q_group].0[q_idx] + new_acc_holder.errs[q_group].0[q_idx] * acc_gamma + acc_2.acc_errs[q_group].0[q_idx] * acc_gamma_sq;
    let l_term_g1 =
      acc_1.acc_errs[l_group].0[l_idx] + new_acc_holder.errs[l_group].0[l_idx] * acc_gamma + acc_2.acc_errs[l_group].0[l_idx] * acc_gamma_sq;
    let c_term_g1 =
      acc_1.acc_errs[c_group].0[c_idx] + new_acc_holder.errs[c_group].0[c_idx] * acc_gamma + acc_2.acc_errs[c_group].0[c_idx] * acc_gamma_sq;
    let term_gt =
      acc_1.acc_errs[gt_group].3[gt_idx] + new_acc_holder.errs[gt_group].3[gt_idx] * acc_gamma + acc_2.acc_errs[gt_group].3[gt_idx] * acc_gamma_sq;
    new_acc_holder.acc_errs = vec![(vec![q_term_g1, l_term_g1, c_term_g1], vec![], vec![], vec![term_gt])];

    new_acc_holder
  }

  fn mira_verify(
    &self,
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    let mut result = true;
    let acc_1_g1 = MatMulG1Terms::<G1Affine>::from_vec(&acc_1.acc_g1);
    let acc_2_g1 = MatMulG1Terms::<G1Affine>::from_vec(&acc_2.acc_g1);
    // Fiat-Shamir
    let mut bytes = Vec::new();

    let acc_1_g1_fiat_shamir = vec![
      acc_1_g1.Left_x,
      acc_1_g1.Left_Q_x,
      acc_1_g1.Left_zero,
      acc_1_g1.Left_zero_div,
      acc_1_g1.Right_x,
      acc_1_g1.Right_Q_x,
      acc_1_g1.Right_zero_div,
      acc_1_g1.Flat_A,
      acc_1_g1.Flat_B,
      acc_1_g1.Flat_C,
    ];
    acc_1_g1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc_1.acc_g2.serialize_uncompressed(&mut bytes).unwrap();

    let acc_2_g1_fiat_shamir = vec![
      acc_2_g1.Left_x,
      acc_2_g1.Left_Q_x,
      acc_2_g1.Left_zero,
      acc_2_g1.Left_zero_div,
      acc_2_g1.Right_x,
      acc_2_g1.Right_Q_x,
      acc_2_g1.Right_zero_div,
      acc_2_g1.Flat_A,
      acc_2_g1.Flat_B,
      acc_2_g1.Flat_C,
    ];
    acc_2_g1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    new_acc.errs.iter().for_each(|(g1, g2, f, gt)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
      gt.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);
    let acc_gamma_sq = acc_gamma * acc_gamma;

    acc_2.acc_g1.iter().enumerate().for_each(|(i, x)| {
      if i < 7 || i >= 11 {
        // Skip the blinding terms
        let z = *x * acc_gamma + acc_1.acc_g1[i];
        result &= new_acc.acc_g1[i] == z;
      }
    });
    result &= new_acc.acc_g2[0] == acc_1.acc_g2[0] + acc_2.acc_g2[0] * acc_gamma;
    result &= new_acc.acc_g2[1] == acc_1.acc_g2[1];
    result &= new_acc.mu == acc_1.mu + acc_gamma * acc_2.mu;
    new_acc.errs[0].0.iter().zip(acc_1.acc_errs[0].0.iter()).enumerate().for_each(|(j, (x, y))| {
      let z = *y + *x * acc_gamma + acc_2.acc_errs[0].0[j] * acc_gamma_sq;
      result &= z == new_acc.acc_errs[0].0[j];
    });
    result &= acc_1.acc_errs[0].3[0] + new_acc.errs[0].3[0] * acc_gamma + acc_2.acc_errs[0].3[0] * acc_gamma_sq == new_acc.acc_errs[0].3[0];
    Some(result)
  }
}

fn index<'a, T>(A: &'a ArrayD<T>, i: usize) -> &'a T {
  if i == 0 {
    A.first().unwrap()
  } else {
    &A[i]
  }
}

#[derive(Debug)]
pub struct MatMulBasicBlock {
  pub m: usize,
  pub n: usize,
}
impl BasicBlock for MatMulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(
      inputs.len() == 2
        && inputs[1].ndim() == 2
        && ((inputs[0].ndim() == 1 && inputs[0].shape()[0] == inputs[1].shape()[1])
          || (inputs[0].ndim() == 2 && inputs[0].shape()[1] == inputs[1].shape()[1]))
    );
    let b = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
    let m = b.shape()[0];
    let n = b.shape()[1];
    if inputs[0].ndim() == 1 {
      let a = inputs[0].view().into_dimensionality::<Ix1>().unwrap();
      let idx_arr = (0..m).collect::<Vec<_>>();
      Ok(vec![arr1(
        &(util::vec_iter(&idx_arr).map(|&i| (0..n).map(|j| a[j] * b[[i, j]]).sum()).collect::<Vec<_>>()),
      )
      .into_dyn()])
    } else {
      let a = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
      let l = a.shape()[0];
      let idx_arr = (0..l * m).collect::<Vec<_>>();
      let res: Vec<_> = util::vec_iter(&idx_arr)
        .map(|idx| {
          let (i, j) = (idx / m, idx % m);
          (0..n).map(|k| a[[i, k]] * b[[j, k]]).sum()
        })
        .collect();
      Ok(vec![ArrayD::from_shape_vec(vec![l, m], res).unwrap()])
    }
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let l = inputs[0].len();
    let m = inputs[0].first().unwrap().raw.len();
    let n = inputs[1].len();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let (alpha, beta) = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("matmul_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      let alpha = alpha.clone();
      let CacheValues::RLCRandom(beta) = cache.entry("matmul_beta".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      (alpha, beta.clone())
    };

    let (alpha_pow, beta_pow) = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::Data(alpha_pow) =
        cache.entry(format!("matmul_alpha_msm_{l}")).or_insert_with(|| CacheValues::Data(Data::new(srs, &calc_pow(alpha, l))))
      else {
        panic!("Cache type error")
      };
      let alpha_pow = alpha_pow.clone();
      let CacheValues::Data(beta_pow) =
        cache.entry(format!("matmul_beta_msm_{n}")).or_insert_with(|| CacheValues::Data(Data::new(srs, &calc_pow(beta, n))))
      else {
        panic!("Cache type error")
      };
      (alpha_pow, beta_pow.clone())
    };

    let mut flat_A = vec![Fr::zero(); m];
    let mut flat_A_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        flat_A[j] += index(inputs[0], i).raw[j] * alpha_pow.raw[i];
      }
      flat_A_r += index(inputs[0], i).r * alpha_pow.raw[i];
    }
    let mut flat_A = Data::new(srs, &flat_A);
    flat_A.r = flat_A_r;

    let mut flat_B = vec![Fr::zero(); m];
    let mut flat_B_r = Fr::zero();
    for i in 0..n {
      for j in 0..m {
        flat_B[j] += inputs[1][i].raw[j] * beta_pow.raw[i];
      }
      flat_B_r += inputs[1][i].r * beta_pow.raw[i];
    }
    let mut flat_B = Data::new(srs, &flat_B);
    flat_B.r = flat_B_r;

    let mut flat_C = vec![Fr::zero(); n];
    let mut flat_C_r = Fr::zero();
    for i in 0..l {
      for j in 0..n {
        flat_C[j] += index(outputs[0], i).raw[j] * alpha_pow.raw[i];
      }
      flat_C_r += index(outputs[0], i).r * alpha_pow.raw[i];
    }
    let mut flat_C = Data::new(srs, &flat_C);
    flat_C.r = flat_C_r;

    // Calculate Left
    let left_raw: Vec<Fr> = (0..m).map(|i| flat_A.raw[i] * flat_B.raw[i]).collect();
    let left_poly = DensePolynomial::from_coefficients_vec(domain_m.ifft(&left_raw));
    let left_x = util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs);
    let left_Q_poly = flat_A.poly.mul(&flat_B.poly).sub(&left_poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let left_Q_x = util::msm::<G1Projective>(&srs.X1A, &left_Q_poly.coeffs);
    let left_zero = srs.X1A[0] * (Fr::from(m as u32).inverse().unwrap() * left_raw.iter().sum::<Fr>());
    let left_zero_div = if left_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs[1..])
    };
    let flat_B_g2 = util::msm::<G2Projective>(&srs.X2A, &flat_B.poly.coeffs) + srs.Y2P * flat_B.r;

    // Calculate Right
    let right_raw: Vec<Fr> = (0..n).map(|i| flat_C.raw[i] * beta_pow.raw[i]).collect();
    let right_poly = DensePolynomial::from_coefficients_vec(domain_n.ifft(&right_raw));
    let right_x = util::msm::<G1Projective>(&srs.X1A, &right_poly.coeffs);
    let right_Q_poly = flat_C.poly.mul(&beta_pow.poly).sub(&right_poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let right_Q_x = util::msm::<G1Projective>(&srs.X1A, &right_Q_poly.coeffs);
    let right_zero_div = if right_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &right_poly.coeffs[1..])
    };

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..7).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let part_corr1 = -(srs.X1P[m] - srs.X1P[0]) * r[1] - srs.X1P[0] * r[0];
    let mut corr = vec![
      part_corr1 + flat_A.g1 * flat_B.r + flat_B.g1 * flat_A.r + srs.Y1P * flat_A.r * flat_B.r,
      -srs.X1P[1] * r[3] + srs.X1P[0] * (r[0] - r[2]),
      -(srs.X1P[n] - srs.X1P[0]) * r[5] + beta_pow.g1 * flat_C.r - srs.X1P[0] * r[4],
      -srs.X1P[1] * r[6] + srs.X1P[0] * (r[4] - r[2] * Fr::from(m as u32) * Fr::from(n as u32).inverse().unwrap()),
    ];
    proof.append(&mut corr);
    let mut proof2 = vec![flat_B_g2];
    let mut fr = vec![];
    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![
        flat_A.g1 + srs.Y1P * flat_A.r,
        flat_B.g1 + srs.Y1P * flat_B.r,
        flat_C.g1 + srs.Y1P * flat_C.r,
        part_corr1,
        flat_A.g1,
        flat_B.g1,
      ];
      let beta_pow_g2 = {
        let mut cache = cache.lock().unwrap();
        let CacheValues::G2(beta_pow_g2) = cache
          .entry(format!("matmul_beta_msm_g2_{n}"))
          .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &beta_pow.poly.coeffs).into()))
        else {
          panic!("Cache type error")
        };
        beta_pow_g2.clone()
      };
      proof.append(&mut additional_g1_for_acc);
      proof2.push(beta_pow_g2.into());
      fr.push(flat_A.r);
      fr.push(flat_B.r);
    }

    return (proof, proof2, fr);
  }

  #[cfg(not(feature = "fold"))]
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let l = inputs[0].len();
    let m = inputs[0].first().unwrap().len;
    let n = inputs[1].len();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let [left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div, corr1, corr2, corr3, corr4] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let flat_B_g2 = proof.1[0];

    let (alpha, beta) = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("matmul_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      let alpha = alpha.clone();
      let CacheValues::RLCRandom(beta) = cache.entry("matmul_beta".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      (alpha, beta.clone())
    };

    let alpha_pow = calc_pow(alpha, l);
    let beta_pow = calc_pow(beta, n);
    let beta_pow_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(beta_pow_g2) = cache
        .entry(format!("matmul_beta_msm_g2_{n}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_n.ifft(&beta_pow)).into()))
      else {
        panic!("Cache type error")
      };
      beta_pow_g2.clone()
    };

    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| index(inputs[0], i).g1).collect();
    let flat_A_g1 = util::msm::<G1Projective>(&temp, &alpha_pow).into();

    // Calculate flat_B
    let temp: Vec<_> = (0..n).map(|i| inputs[1][i].g1).collect();
    let flat_B_g1 = util::msm::<G1Projective>(&temp, &beta_pow).into();

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| index(outputs[0], i).g1).collect();
    let flat_C_g1 = util::msm::<G1Projective>(&temp, &alpha_pow).into();

    // Check left(x) (left_i = flat_A_i * flat_B_i)
    checks.push(vec![
      (flat_A_g1, flat_B_g2),
      (-left_x, srs.X2A[0]),
      (-left_Q_x, (srs.X2A[m] - srs.X2A[0]).into()),
      (-corr1, srs.Y2A),
    ]);

    // Check flat_B_g2
    checks.push(vec![(flat_B_g1, srs.X2A[0]), (srs.X1A[0], -flat_B_g2)]);

    // Check left(x) - left(0) is divisible by x
    checks.push(vec![
      ((left_x - left_zero).into(), srs.X2A[0]),
      (-left_zero_div, srs.X2A[1]),
      (-corr2, srs.Y2A),
    ]);

    // Check right(x) (right_i = flat_C_i * beta_pow_i)
    checks.push(vec![
      (flat_C_g1, beta_pow_g2),
      (-right_x, srs.X2A[0]),
      (-right_Q_x, (srs.X2A[n] - srs.X2A[0]).into()),
      (-corr3, srs.Y2A),
    ]);

    // Assume right(0) = left(0)*n/m (which assumes ∑left=∑right)
    let right_zero: G1Affine = (left_zero * (Fr::from(m as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // check right(x) - right(0) is divisible by x
    checks.push(vec![
      ((right_x - right_zero).into(), srs.X2A[0]),
      (-right_zero_div, srs.X2A[1]),
      (-corr4, srs.Y2A),
    ]);

    checks
  }

  #[cfg(feature = "fold")]
  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let (_alpha, _beta) = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("matmul_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      let alpha = alpha.clone();
      let CacheValues::RLCRandom(beta) = cache.entry("matmul_beta".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      (alpha, beta.clone())
    };
    vec![]
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: AccProofProjRef,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> AccProofProj {
    let proof = self.prover_proof_to_acc(proof);
    if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
      return holder_to_acc_proof(proof);
    }
    let acc_proof = acc_proof_to_holder(self, acc_proof, true);
    holder_to_acc_proof(self.mira_prove(srs, acc_proof, proof, rng))
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: AccProofProjRef,
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), AccProofAffine) {
    let mut acc_holder = acc_proof_to_holder(self, acc_proof, true);

    let mut acc_g1 = MatMulG1Terms::<G1Projective>::from_vec(&acc_holder.acc_g1);
    let acc_fr = MatMulFrTerms::<Fr>::from_vec(&acc_holder.acc_fr);
    acc_g1.Corr1 = acc_g1.Part_corr1.unwrap() * acc_holder.mu
      + acc_g1.Flat_A_no_blind.unwrap() * acc_fr.Flat_B_r.unwrap()
      + acc_g1.Flat_B_no_blind.unwrap() * acc_fr.Flat_A_r.unwrap()
      + srs.Y1P * acc_fr.Flat_A_r.unwrap() * acc_fr.Flat_B_r.unwrap();
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_g1.to_vec()[..MatMulG1Terms::<G1Projective>::PUBLIC_COUNT].to_vec();
    acc_holder.acc_fr = vec![];
    let acc_proof = holder_to_acc_proof(acc_holder);

    // remove blinding terms from bb proof for the verifier
    let cqlin_proof = (proof.0[..MatMulG1Terms::<G1Projective>::PUBLIC_COUNT].to_vec(), proof.1.to_vec(), vec![]);

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
        acc_proof.3.iter().map(|x| *x).collect(),
      ),
    )
  }

  fn acc_verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: AccProofAffineRef,
    acc_proof: AccProofAffineRef,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Option<bool> {
    let mut result = true;

    let l = inputs[0].len();
    let m = inputs[0].first().unwrap().len;
    let n = inputs[1].len();
    assert!(m == self.m && n == self.n);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let (alpha, beta) = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("matmul_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      let alpha = alpha.clone();
      let CacheValues::RLCRandom(beta) = cache.entry("matmul_beta".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      (alpha, beta.clone())
    };

    let alpha_pow = calc_pow(alpha, l);
    let beta_pow = calc_pow(beta, n);
    let beta_pow_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(beta_pow_g2) = cache
        .entry(format!("matmul_beta_msm_g2_{n}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_n.ifft(&beta_pow)).into()))
      else {
        panic!("Cache type error")
      };
      beta_pow_g2.clone()
    };
    result &= beta_pow_g2 == proof.1[1];

    // Calculate flat_A
    let temp: Vec<_> = (0..l).map(|i| index(inputs[0], i).g1).collect();
    let flat_A = util::msm::<G1Projective>(&temp, &alpha_pow);

    // Calculate flat_B
    let temp: Vec<_> = (0..n).map(|i| inputs[1][i].g1).collect();
    let flat_B = util::msm::<G1Projective>(&temp, &beta_pow);

    // Calculate flat_C
    let temp: Vec<_> = (0..l).map(|i| index(outputs[0], i).g1).collect();
    let flat_C = util::msm::<G1Projective>(&temp, &alpha_pow);

    let proof_g1 = MatMulG1Terms::<G1Affine>::from_vec(&proof.0);
    result &= flat_A == proof_g1.Flat_A && flat_B == proof_g1.Flat_B;
    result &= flat_C == proof_g1.Flat_C;
    if prev_acc_proof.2.len() == 0 && acc_proof.2[0].is_one() {
      return Some(result);
    }
    let proof = self.verifier_proof_to_acc(proof);
    let prev_acc_holder = acc_proof_to_holder(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_holder(self, acc_proof, false);
    result &= self.mira_verify(prev_acc_holder, proof, acc_holder, rng).unwrap();
    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: AccProofAffineRef) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let acc_holder = acc_proof_to_holder(self, acc_proof, false);
    let acc_g1 = MatMulG1Terms::<G1Affine>::from_vec(&acc_holder.acc_g1);
    let acc_g2 = MatMulG2Terms::<G2Affine>::from_vec(&acc_holder.acc_g2);

    let acc_mu = acc_holder.mu;
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    temp.push((
      err_1.0[MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_Q).1],
      (srs.X2A[self.m] - srs.X2A[0]).into(),
    ));
    temp.push((err_1.0[MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_l).1], srs.X2A[0]));
    temp.push((err_1.0[MatMulErrG1Terms::idx(MatMulErrG1Terms::Err_C).1], srs.Y2A));

    let mut acc_1: PairingCheck = vec![
      (acc_g1.Flat_A, acc_g2.Flat_B_g2),
      ((-acc_g1.Left_x * acc_mu).into(), srs.X2A[0]),
      ((-acc_g1.Left_Q_x * acc_mu).into(), (srs.X2A[self.m] - srs.X2A[0]).into()),
      (-acc_g1.Corr1, srs.Y2A),
    ];
    acc_1.extend(temp);

    let acc_2: PairingCheck = vec![(acc_g1.Flat_B, srs.X2A[0]), (srs.X1A[0], -acc_g2.Flat_B_g2)];

    let acc_3: PairingCheck = vec![
      ((acc_g1.Left_x - acc_g1.Left_zero).into(), srs.X2A[0]),
      (-acc_g1.Left_zero_div, srs.X2A[1]),
      (-acc_g1.Corr2, srs.Y2A),
    ];

    let acc_4: PairingCheck = vec![
      (acc_g1.Flat_C, acc_g2.Beta_pow_g2),
      (-acc_g1.Right_x, srs.X2A[0]),
      (-acc_g1.Right_Q_x, (srs.X2A[self.n] - srs.X2A[0]).into()),
      (-acc_g1.Corr3, srs.Y2A),
    ];

    let acc_right_zero: G1Projective = acc_g1.Left_zero * (Fr::from(self.m as u32) * Fr::from(self.n as u32).inverse().unwrap());
    let acc_5 = vec![
      ((-acc_right_zero + acc_g1.Right_x).into(), srs.X2A[0]),
      (-acc_g1.Right_zero_div, srs.X2A[1]),
      (-acc_g1.Corr4, srs.Y2A),
    ];

    let pairing_zero = PairingOutput::<Bn<ark_bn254::Config>>::zero();
    vec![
      (acc_1, err_1.3[MatMulErrGtTerms::idx(MatMulErrGtTerms::Err_Gt).1]),
      (acc_2, pairing_zero),
      (acc_3, pairing_zero),
      (acc_4, pairing_zero),
      (acc_5, pairing_zero),
    ]
  }

  fn acc_finalize(&self, _srs: &SRS, acc_proof: AccProofAffineRef) -> AccProofAffine {
    let mut acc_holder = acc_proof_to_holder(self, acc_proof, false);
    let err_1 = &acc_holder.acc_errs[0];
    let acc_err1 = err_1.3[0].clone();
    acc_holder.errs = vec![];
    acc_holder.acc_errs = vec![];
    let acc_proof = holder_to_acc_proof(acc_holder);
    (acc_proof.0, acc_proof.1, acc_proof.2, vec![acc_err1])
  }
}
