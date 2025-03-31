#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_to_acc_proof, calc_pow, AccHolder};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use ndarray::{arr1, arr2, ArrayD, Ix1, Ix2, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::ParallelIterator;

struct MatMulAccProof<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize> {
  fiat_shamir: MatMulAccFiatShamir<P, Q>,
  acc_corr: [P; 4],
  mu: Fr,
  prover_only: Option<MatMulAccProofProverOnly<P>>,
  errs: MatMulErrs<P, Q>,
  acc_errs: MatMulAccErrs<P, Q>,
}

#[derive(CanonicalSerialize)]
struct MatMulAccFiatShamir<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize> {
  acc_left_x: P,
  acc_left_Q_x: P,
  acc_left_zero: P,
  acc_left_zero_div: P,
  acc_right_x: P,
  acc_right_Q_x: P,
  acc_right_zero_div: P,
  acc_flat_A: P,
  acc_flat_B: P,
  acc_flat_C: P,
  acc_flat_B_g2: Q,
  acc_beta_pow_g2: Q,
}

struct MatMulAccProofProverOnly<P: Copy + CanonicalSerialize> {
  acc_part_corr1: P,
  acc_flat_A_no_blind: P,
  acc_flat_B_no_blind: P,
  acc_flat_A_r: Fr,
  acc_flat_B_r: Fr,
}

#[derive(CanonicalSerialize, Clone)]
struct MatMulErrs<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize> {
  flat_A: P,
  acc_flat_A: P,
  acc_left_Q_x: P,
  acc_left_x: P,
  acc_part_corr1: P,
  acc_flat_B_g2: Q,
  flat_B_g2: Q,
}

#[derive(Clone)]
struct MatMulAccErrs<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize> {
  prev_g1s: Vec<P>,
  flat_A: P,
  acc_flat_A: P,
  acc_left_Q_x: P,
  acc_left_x: P,
  acc_part_corr1: P,
  prev_g2s: Vec<Q>,
  acc_flat_B_g2: Q,
  flat_B_g2: Q,
}

fn accumulate(
  matmul_acc: &MatMulAccProof<G1Projective, G2Projective>,
  errs: &MatMulErrs<G1Projective, G2Projective>,
  proof: &(&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  acc_gamma: Fr,
) -> MatMulAccProof<G1Projective, G2Projective> {
  let [left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div, corr1, corr2, corr3, corr4, flat_A, flat_B, flat_C, part_corr1, flat_A_no_blind, flat_B_no_blind] =
    proof.0[..]
  else {
    panic!("Wrong proof format")
  };

  let [flat_B_g2, beta_pow_g2] = proof.1[..] else {
    panic!("Wrong proof format")
  };
  let [flat_A_r, flat_B_r] = proof.2[..] else {
    panic!("Wrong proof format")
  };

  let new_errs = MatMulErrs {
    flat_A: errs.flat_A * acc_gamma,
    acc_flat_A: errs.acc_flat_A * acc_gamma,
    acc_left_Q_x: errs.acc_left_Q_x * acc_gamma,
    acc_left_x: errs.acc_left_x * acc_gamma,
    acc_part_corr1: errs.acc_part_corr1 * acc_gamma,
    acc_flat_B_g2: errs.acc_flat_B_g2,
    flat_B_g2: errs.flat_B_g2,
  };

  let mut acc_errs = MatMulAccErrs {
    prev_g1s: matmul_acc.acc_errs.prev_g1s.clone(),
    flat_A: new_errs.flat_A,
    acc_flat_A: new_errs.acc_flat_A,
    acc_left_Q_x: matmul_acc.acc_errs.acc_left_Q_x + new_errs.acc_left_Q_x,
    acc_left_x: matmul_acc.acc_errs.acc_left_x + new_errs.acc_left_x,
    acc_part_corr1: matmul_acc.acc_errs.acc_part_corr1 + new_errs.acc_part_corr1,
    prev_g2s: matmul_acc.acc_errs.prev_g2s.clone(),
    acc_flat_B_g2: new_errs.acc_flat_B_g2,
    flat_B_g2: new_errs.flat_B_g2,
  };
  acc_errs.prev_g1s.push(matmul_acc.acc_errs.flat_A);
  acc_errs.prev_g1s.push(matmul_acc.acc_errs.acc_flat_A);
  acc_errs.prev_g2s.push(matmul_acc.acc_errs.acc_flat_B_g2);
  acc_errs.prev_g2s.push(matmul_acc.acc_errs.flat_B_g2);

  // Compute the error
  let matmul_acc_prover_only = matmul_acc.prover_only.as_ref().unwrap();
  let new_matmul_acc = MatMulAccProof {
    fiat_shamir: MatMulAccFiatShamir {
      acc_left_x: matmul_acc.fiat_shamir.acc_left_x + left_x * acc_gamma,
      acc_left_Q_x: matmul_acc.fiat_shamir.acc_left_Q_x + left_Q_x * acc_gamma,
      acc_left_zero: matmul_acc.fiat_shamir.acc_left_zero + left_zero * acc_gamma,
      acc_left_zero_div: matmul_acc.fiat_shamir.acc_left_zero_div + left_zero_div * acc_gamma,
      acc_right_x: matmul_acc.fiat_shamir.acc_right_x + right_x * acc_gamma,
      acc_right_Q_x: matmul_acc.fiat_shamir.acc_right_Q_x + right_Q_x * acc_gamma,
      acc_right_zero_div: matmul_acc.fiat_shamir.acc_right_zero_div + right_zero_div * acc_gamma,
      acc_flat_A: matmul_acc.fiat_shamir.acc_flat_A + flat_A * acc_gamma,
      acc_flat_B: matmul_acc.fiat_shamir.acc_flat_B + flat_B * acc_gamma,
      acc_flat_C: matmul_acc.fiat_shamir.acc_flat_C + flat_C * acc_gamma,
      acc_flat_B_g2: matmul_acc.fiat_shamir.acc_flat_B_g2 + flat_B_g2 * acc_gamma,
      acc_beta_pow_g2: beta_pow_g2,
    },
    acc_corr: [
      matmul_acc.acc_corr[0] + corr1 * acc_gamma,
      matmul_acc.acc_corr[1] + corr2 * acc_gamma,
      matmul_acc.acc_corr[2] + corr3 * acc_gamma,
      matmul_acc.acc_corr[3] + corr4 * acc_gamma,
    ],
    mu: matmul_acc.mu + acc_gamma,
    prover_only: Some(MatMulAccProofProverOnly {
      acc_part_corr1: matmul_acc_prover_only.acc_part_corr1 + part_corr1 * acc_gamma,
      acc_flat_A_no_blind: matmul_acc_prover_only.acc_flat_A_no_blind + flat_A_no_blind * acc_gamma,
      acc_flat_B_no_blind: matmul_acc_prover_only.acc_flat_B_no_blind + flat_B_no_blind * acc_gamma,
      acc_flat_A_r: matmul_acc_prover_only.acc_flat_A_r + flat_A_r * acc_gamma,
      acc_flat_B_r: matmul_acc_prover_only.acc_flat_B_r + flat_B_r * acc_gamma,
    }),
    errs: errs.clone(),
    acc_errs,
  };

  new_matmul_acc
}

fn acc_proof_to_matmul_acc_holder<P: Copy, Q: Copy>(acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>), is_prover: bool) -> AccHolder<P, Q> {
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

  // [acc_left_x, acc_left_Q_x, acc_left_zero, acc_left_zero_div, acc_right_x,
  // acc_right_Q_x, acc_right_zero_div, acc_corr1, acc_corr2, acc_corr3,
  // acc_corr4, acc_flat_A, acc_flat_B, acc_flat_C, acc_part_corr1,
  // acc_flat_A_no_blind, acc_flat_B_no_blind]
  let acc_g1_num = if is_prover { 17 } else { 14 };
  let acc_fr_num = if is_prover { 2 } else { 0 };
  let acc_err_g2_num = acc_proof.1.len() - 4;

  let err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[acc_g1_num..(acc_g1_num + 5)].to_vec(), acc_proof.1[2..4].to_vec(), vec![]);

  let errs = vec![err1];

  let acc_err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 5)..(acc_g1_num + 8 + acc_err_g2_num)].to_vec(),
    acc_proof.1[4..(4 + acc_err_g2_num)].to_vec(),
    vec![],
  );

  let acc_errs = vec![acc_err1];

  AccHolder {
    acc_g1: acc_proof.0[..acc_g1_num].to_vec(),
    acc_g2: acc_proof.1[..2].to_vec(),
    acc_fr: acc_proof.2[..acc_fr_num].to_vec(),
    mu: acc_proof.2[acc_proof.2.len() - 1],
    errs,
    acc_errs,
  }
}

fn acc_proof_to_matmul_acc<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize>(
  acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>),
  is_prover: bool,
) -> Option<MatMulAccProof<P, Q>> {
  if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
    return None;
  }
  let acc_holder = acc_proof_to_matmul_acc_holder(acc_proof, is_prover);
  Some(matmul_acc_holder_to_acc(acc_holder, is_prover))
}

fn matmul_acc_holder_to_acc<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize>(
  acc_holder: AccHolder<P, Q>,
  is_prover: bool,
) -> MatMulAccProof<P, Q> {
  let g1_len = acc_holder.acc_errs[0].0.len();
  let g2_len = acc_holder.acc_errs[0].1.len();
  let acc_proof = MatMulAccProof {
    fiat_shamir: MatMulAccFiatShamir {
      acc_left_x: acc_holder.acc_g1[0],
      acc_left_Q_x: acc_holder.acc_g1[1],
      acc_left_zero: acc_holder.acc_g1[2],
      acc_left_zero_div: acc_holder.acc_g1[3],
      acc_right_x: acc_holder.acc_g1[4],
      acc_right_Q_x: acc_holder.acc_g1[5],
      acc_right_zero_div: acc_holder.acc_g1[6],
      acc_flat_A: acc_holder.acc_g1[11],
      acc_flat_B: acc_holder.acc_g1[12],
      acc_flat_C: acc_holder.acc_g1[13],
      acc_flat_B_g2: acc_holder.acc_g2[0],
      acc_beta_pow_g2: acc_holder.acc_g2[1],
    },
    acc_corr: [acc_holder.acc_g1[7], acc_holder.acc_g1[8], acc_holder.acc_g1[9], acc_holder.acc_g1[10]],
    mu: acc_holder.mu,
    prover_only: if is_prover {
      Some(MatMulAccProofProverOnly {
        acc_part_corr1: acc_holder.acc_g1[14],
        acc_flat_A_no_blind: acc_holder.acc_g1[15],
        acc_flat_B_no_blind: acc_holder.acc_g1[16],
        acc_flat_A_r: acc_holder.acc_fr[0],
        acc_flat_B_r: acc_holder.acc_fr[1],
      })
    } else {
      None
    },
    errs: MatMulErrs {
      flat_A: acc_holder.errs[0].0[0],
      acc_flat_A: acc_holder.errs[0].0[1],
      acc_left_Q_x: acc_holder.errs[0].0[2],
      acc_left_x: acc_holder.errs[0].0[3],
      acc_part_corr1: acc_holder.errs[0].0[4],
      acc_flat_B_g2: acc_holder.errs[0].1[0],
      flat_B_g2: acc_holder.errs[0].1[1],
    },
    acc_errs: MatMulAccErrs {
      prev_g1s: acc_holder.acc_errs[0].0[..g1_len - 5].to_vec(),
      flat_A: acc_holder.acc_errs[0].0[g1_len - 5],
      acc_flat_A: acc_holder.acc_errs[0].0[g1_len - 4],
      acc_left_Q_x: acc_holder.acc_errs[0].0[g1_len - 3],
      acc_left_x: acc_holder.acc_errs[0].0[g1_len - 2],
      acc_part_corr1: acc_holder.acc_errs[0].0[g1_len - 1],
      prev_g2s: acc_holder.acc_errs[0].1[..g2_len - 2].to_vec(),
      acc_flat_B_g2: acc_holder.acc_errs[0].1[g2_len - 2],
      flat_B_g2: acc_holder.acc_errs[0].1[g2_len - 1],
    },
  };
  acc_proof
}

fn matmul_acc_to_acc_holder<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize>(
  acc: MatMulAccProof<P, Q>,
  is_prover: bool,
) -> AccHolder<P, Q> {
  let acc_g2 = vec![acc.fiat_shamir.acc_flat_B_g2, acc.fiat_shamir.acc_beta_pow_g2];
  let mu = acc.mu;
  let errs = vec![(
    vec![
      acc.errs.flat_A,
      acc.errs.acc_flat_A,
      acc.errs.acc_left_Q_x,
      acc.errs.acc_left_x,
      acc.errs.acc_part_corr1,
    ],
    vec![acc.errs.acc_flat_B_g2, acc.errs.flat_B_g2],
    vec![],
  )];
  let acc_errs = vec![(
    vec![
      acc.acc_errs.flat_A,
      acc.acc_errs.acc_flat_A,
      acc.acc_errs.acc_left_Q_x,
      acc.acc_errs.acc_left_x,
      acc.acc_errs.acc_part_corr1,
    ],
    vec![acc.acc_errs.acc_flat_B_g2, acc.acc_errs.flat_B_g2],
    vec![],
  )];
  if is_prover {
    let prover_only = acc.prover_only.unwrap();
    let acc_g1 = vec![
      acc.fiat_shamir.acc_left_x,
      acc.fiat_shamir.acc_left_Q_x,
      acc.fiat_shamir.acc_left_zero,
      acc.fiat_shamir.acc_left_zero_div,
      acc.fiat_shamir.acc_right_x,
      acc.fiat_shamir.acc_right_Q_x,
      acc.fiat_shamir.acc_right_zero_div,
      acc.acc_corr[0],
      acc.acc_corr[1],
      acc.acc_corr[2],
      acc.acc_corr[3],
      acc.fiat_shamir.acc_flat_A,
      acc.fiat_shamir.acc_flat_B,
      acc.fiat_shamir.acc_flat_C,
      prover_only.acc_part_corr1,
      prover_only.acc_flat_A_no_blind,
      prover_only.acc_flat_B_no_blind,
    ];
    let acc_fr = vec![prover_only.acc_flat_A_r, prover_only.acc_flat_B_r];
    AccHolder {
      acc_g1,
      acc_g2,
      acc_fr,
      mu,
      errs,
      acc_errs,
    }
  } else {
    let acc_g1 = vec![
      acc.fiat_shamir.acc_left_x,
      acc.fiat_shamir.acc_left_Q_x,
      acc.fiat_shamir.acc_left_zero,
      acc.fiat_shamir.acc_left_zero_div,
      acc.fiat_shamir.acc_right_x,
      acc.fiat_shamir.acc_right_Q_x,
      acc.fiat_shamir.acc_right_zero_div,
      acc.acc_corr[0],
      acc.acc_corr[1],
      acc.acc_corr[2],
      acc.acc_corr[3],
      acc.fiat_shamir.acc_flat_A,
      acc.fiat_shamir.acc_flat_B,
      acc.fiat_shamir.acc_flat_C,
    ];
    AccHolder {
      acc_g1,
      acc_g2,
      acc_fr: vec![],
      mu,
      errs,
      acc_errs,
    }
  }
}

fn matmul_acc_to_acc_proof<P: Copy + CanonicalSerialize, Q: Copy + CanonicalSerialize>(
  acc: MatMulAccProof<P, Q>,
  is_prover: bool,
) -> (Vec<P>, Vec<Q>, Vec<Fr>) {
  let acc_holder = matmul_acc_to_acc_holder(acc, is_prover);
  acc_to_acc_proof(acc_holder)
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

  fn acc_init(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
    proof.0[11..14].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);

    // acc errs and errs
    let g1_zero = G1Projective::zero();
    let g2_zero = G2Projective::zero();
    acc_proof.0.extend(vec![g1_zero; 5 * 2]);
    acc_proof.1.extend(vec![g2_zero; 2 * 2]);

    // mu
    acc_proof.2.push(Fr::one());
    acc_proof
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let [left_x, left_Q_x, _left_zero, _left_zero_div, _right_x, _right_Q_x, _right_zero_div, _corr1, _corr2, _corr3, _corr4, flat_A, _flat_B, _flat_C, part_corr1, flat_A_no_blind, flat_B_no_blind] =
      proof.0[..]
    else {
      panic!("Wrong proof format")
    };

    let [flat_B_g2, beta_pow_g2] = proof.1[..] else {
      panic!("Wrong proof format")
    };
    let [flat_A_r, flat_B_r] = proof.2[..] else {
      panic!("Wrong proof format")
    };

    let matmul_acc = acc_proof_to_matmul_acc(acc_proof, true).unwrap();

    let acc_flat_A = matmul_acc.fiat_shamir.acc_flat_A;
    let acc_left_x = matmul_acc.fiat_shamir.acc_left_x;
    let acc_left_Q_x = matmul_acc.fiat_shamir.acc_left_Q_x;
    let acc_mu = matmul_acc.mu;
    let acc_part_corr1 = matmul_acc.prover_only.as_ref().unwrap().acc_part_corr1;
    let po = matmul_acc.prover_only.as_ref().unwrap();
    let acc_flat_A_no_blind = po.acc_flat_A_no_blind;
    let acc_flat_B_no_blind = po.acc_flat_B_no_blind;
    let acc_flat_A_r = po.acc_flat_A_r;
    let acc_flat_B_r = po.acc_flat_B_r;
    let acc_flat_B_g2 = matmul_acc.fiat_shamir.acc_flat_B_g2;

    let errs = MatMulErrs {
      flat_A,
      acc_flat_A,
      acc_left_Q_x: acc_left_Q_x + left_Q_x * acc_mu,
      acc_left_x: acc_left_x + left_x * acc_mu,
      acc_part_corr1: acc_part_corr1
        + part_corr1 * acc_mu
        + acc_flat_A_no_blind * flat_B_r
        + flat_A_no_blind * acc_flat_B_r
        + acc_flat_B_no_blind * flat_A_r
        + flat_B_no_blind * acc_flat_A_r
        + srs.Y1P * (flat_A_r * acc_flat_B_r + flat_B_r * acc_flat_A_r),
      acc_flat_B_g2,
      flat_B_g2,
    };

    // Fiat-Shamir
    let mut bytes = Vec::new();
    matmul_acc.fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
    proof.0[11..14].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    errs.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let new_matmul_acc = accumulate(&matmul_acc, &errs, &proof, acc_gamma);

    matmul_acc_to_acc_proof(new_matmul_acc, true)
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    let mut matmul_acc = acc_proof_to_matmul_acc(acc_proof, true).unwrap();

    let po = matmul_acc.prover_only.as_ref().unwrap();
    matmul_acc.acc_corr[0] = po.acc_part_corr1 * matmul_acc.mu
      + po.acc_flat_A_no_blind * po.acc_flat_B_r
      + po.acc_flat_B_no_blind * po.acc_flat_A_r
      + srs.Y1P * po.acc_flat_A_r * po.acc_flat_B_r;

    matmul_acc.prover_only = None;
    let acc_proof = matmul_acc_to_acc_proof(matmul_acc, false);

    // remove blinding terms from bb proof for the verifier
    let cqlin_proof = (proof.0[..11].to_vec(), proof.1.to_vec(), vec![]);

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
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
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

    let proof0_11_14 = vec![flat_A, flat_B, flat_C];

    let prev_acc = acc_proof_to_matmul_acc(prev_acc_proof, false);
    let acc = acc_proof_to_matmul_acc(acc_proof, false).unwrap();

    if prev_acc.is_none() || (prev_acc.as_ref().unwrap().mu.is_zero() && acc.mu.is_one()) {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
      proof0_11_14.serialize_uncompressed(&mut bytes).unwrap();
      proof.1.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    let prev_acc = prev_acc.unwrap();
    prev_acc.fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
    proof0_11_14.serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    acc.errs.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    result &= (proof.0[0] * acc_gamma + prev_acc.fiat_shamir.acc_left_x) == acc.fiat_shamir.acc_left_x;
    result &= (proof.0[1] * acc_gamma + prev_acc.fiat_shamir.acc_left_Q_x) == acc.fiat_shamir.acc_left_Q_x;
    result &= (proof.0[2] * acc_gamma + prev_acc.fiat_shamir.acc_left_zero) == acc.fiat_shamir.acc_left_zero;
    result &= (proof.0[3] * acc_gamma + prev_acc.fiat_shamir.acc_left_zero_div) == acc.fiat_shamir.acc_left_zero_div;
    result &= (proof.0[4] * acc_gamma + prev_acc.fiat_shamir.acc_right_x) == acc.fiat_shamir.acc_right_x;
    result &= (proof.0[5] * acc_gamma + prev_acc.fiat_shamir.acc_right_Q_x) == acc.fiat_shamir.acc_right_Q_x;
    result &= (proof.0[6] * acc_gamma + prev_acc.fiat_shamir.acc_right_zero_div) == acc.fiat_shamir.acc_right_zero_div;

    result &= (flat_A * acc_gamma + prev_acc.fiat_shamir.acc_flat_A) == acc.fiat_shamir.acc_flat_A;
    result &= (flat_B * acc_gamma + prev_acc.fiat_shamir.acc_flat_B) == acc.fiat_shamir.acc_flat_B;
    result &= (flat_C * acc_gamma + prev_acc.fiat_shamir.acc_flat_C) == acc.fiat_shamir.acc_flat_C;

    result &= acc.fiat_shamir.acc_flat_B_g2 == prev_acc.fiat_shamir.acc_flat_B_g2 + proof.1[0] * acc_gamma;
    result &= acc.fiat_shamir.acc_beta_pow_g2 == prev_acc.fiat_shamir.acc_beta_pow_g2
      && proof.1[1] == beta_pow_g2
      && beta_pow_g2 == acc.fiat_shamir.acc_beta_pow_g2;
    result &= acc.mu == prev_acc.mu + acc_gamma;

    result &= prev_acc.acc_errs.acc_left_Q_x + acc.errs.acc_left_Q_x * acc_gamma == acc.acc_errs.acc_left_Q_x;
    result &= prev_acc.acc_errs.acc_left_x + acc.errs.acc_left_x * acc_gamma == acc.acc_errs.acc_left_x;
    result &= prev_acc.acc_errs.acc_part_corr1 + acc.errs.acc_part_corr1 * acc_gamma == acc.acc_errs.acc_part_corr1;

    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let acc = acc_proof_to_matmul_acc(acc_proof, false).unwrap();

    let acc_flat_A = acc.fiat_shamir.acc_flat_A;
    let acc_flat_B_g2 = acc.fiat_shamir.acc_flat_B_g2;
    let acc_left_x = acc.fiat_shamir.acc_left_x;
    let acc_left_Q_x = acc.fiat_shamir.acc_left_Q_x;
    let acc_left_zero = acc.fiat_shamir.acc_left_zero;
    let acc_left_zero_div = acc.fiat_shamir.acc_left_zero_div;
    let acc_right_x = acc.fiat_shamir.acc_right_x;
    let acc_right_Q_x = acc.fiat_shamir.acc_right_Q_x;
    let acc_right_zero_div = acc.fiat_shamir.acc_right_zero_div;
    let acc_flat_B = acc.fiat_shamir.acc_flat_B;
    let acc_flat_C = acc.fiat_shamir.acc_flat_C;
    let beta_pow_g2 = acc.fiat_shamir.acc_beta_pow_g2;
    let acc_corr = acc.acc_corr;
    let acc_mu = acc.mu;
    let errs = &acc.acc_errs;

    let mut temp: PairingCheck = vec![];
    for i in 0..errs.prev_g2s.len() {
      temp.push((-errs.prev_g1s[i], errs.prev_g2s[i]));
    }
    temp.push((-errs.flat_A, errs.acc_flat_B_g2));
    temp.push((-errs.acc_flat_A, errs.flat_B_g2));
    temp.push((errs.acc_left_Q_x, (srs.X2A[self.m] - srs.X2A[0]).into()));
    temp.push((errs.acc_left_x, srs.X2A[0]));
    temp.push((errs.acc_part_corr1, srs.Y2A));

    let err_1 = temp;
    let mut acc_1: PairingCheck = vec![
      (acc_flat_A, acc_flat_B_g2),
      ((-acc_left_x * acc_mu).into(), srs.X2A[0]),
      ((-acc_left_Q_x * acc_mu).into(), (srs.X2A[self.m] - srs.X2A[0]).into()),
      (-acc_corr[0], srs.Y2A),
    ];
    acc_1.extend(err_1);

    let acc_2: PairingCheck = vec![(acc_flat_B, srs.X2A[0]), (srs.X1A[0], -acc_flat_B_g2)];

    let acc_3: PairingCheck = vec![
      ((acc_left_x - acc_left_zero).into(), srs.X2A[0]),
      (-acc_left_zero_div, srs.X2A[1]),
      (-acc_corr[1], srs.Y2A),
    ];

    let acc_4: PairingCheck = vec![
      (acc_flat_C, beta_pow_g2),
      (-acc_right_x, srs.X2A[0]),
      (-acc_right_Q_x, (srs.X2A[self.n] - srs.X2A[0]).into()),
      (-acc_corr[2], srs.Y2A),
    ];

    let acc_right_zero: G1Projective = acc_left_zero * (Fr::from(self.m as u32) * Fr::from(self.n as u32).inverse().unwrap());
    let acc_5 = vec![
      ((-acc_right_zero + acc_right_x).into(), srs.X2A[0]),
      (-acc_right_zero_div, srs.X2A[1]),
      (-acc_corr[3], srs.Y2A),
    ];

    vec![acc_1, acc_2, acc_3, acc_4, acc_5]
  }

  fn acc_clean_errs(&self, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) {
    let mut acc_holder = acc_proof_to_matmul_acc_holder(acc_proof, false);
    acc_holder.errs = vec![];
    acc_to_acc_proof(acc_holder)
  }
}
