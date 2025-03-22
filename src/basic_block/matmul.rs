#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_proof_to_acc, acc_to_acc_proof, calc_pow, AccHolder, AccProofLayout};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use ndarray::{arr1, arr2, ArrayD, Ix1, Ix2, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::ParallelIterator;

impl AccProofLayout for MatMulBasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    if is_prover {
      17
    } else {
      14
    }
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    2
  }
  fn acc_fr_num(&self, is_prover: bool) -> usize {
    if is_prover {
      2
    } else {
      0
    }
  }
  fn err_g1_nums_summable(&self) -> Vec<usize> {
    vec![3]
  }
  fn err_g1_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_g2_nums_summable(&self) -> Vec<usize> {
    vec![0]
  }
  fn err_g2_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![0]
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

    let acc_holder = acc_proof_to_acc(self, acc_proof, true);
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };

    let [acc_left_x, acc_left_Q_x, _acc_left_zero, _acc_left_zero_div, _acc_right_x, _acc_right_Q_x, _acc_right_zero_div, _acc_corr1, _acc_corr2, _acc_corr3, _acc_corr4, acc_flat_A, _acc_flat_B, _acc_flat_C, acc_part_corr1, acc_flat_A_no_blind, acc_flat_B_no_blind] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong acc proof format")
    };
    let [acc_flat_B_g2, acc_beta_pow_g2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    assert!(beta_pow_g2 == acc_beta_pow_g2);
    let acc_mu = acc_holder.mu;
    let [acc_flat_A_r, acc_flat_B_r] = acc_holder.acc_fr[..] else {
      panic!("Wrong acc proof format")
    };

    // Compute the error
    let err: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) = (
      vec![
        flat_A,
        acc_flat_A,
        acc_left_Q_x + left_Q_x * acc_mu,
        acc_left_x + left_x * acc_mu,
        acc_part_corr1
          + part_corr1 * acc_mu
          + acc_flat_A_no_blind * flat_B_r
          + flat_A_no_blind * acc_flat_B_r
          + acc_flat_B_no_blind * flat_A_r
          + flat_B_no_blind * acc_flat_A_r
          + srs.Y1P * (flat_A_r * acc_flat_B_r + flat_B_r * acc_flat_A_r),
      ],
      vec![acc_flat_B_g2, flat_B_g2],
      vec![],
    );
    let mut errs = vec![err];

    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_holder.acc_g1[..7].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g1[11..14].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
    proof.0[11..14].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    new_acc_holder.acc_g1 = proof.0.iter().zip(acc_holder.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.acc_g2 = vec![flat_B_g2 * acc_gamma + acc_flat_B_g2, beta_pow_g2];
    new_acc_holder.acc_fr = proof.2.iter().zip(acc_holder.acc_fr.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.mu = acc_mu + acc_gamma;
    new_acc_holder.errs = errs.clone();
    new_acc_holder.acc_errs = acc_holder.acc_errs;

    errs[0].0 = errs[0].0.iter().map(|x| (*x * acc_gamma).into()).collect();

    // Append error terms
    let err1_g1_len = new_acc_holder.acc_errs[0].0.len();
    let q_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 3].clone() + errs[0].0[2];
    let l_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 2].clone() + errs[0].0[3];
    let c_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 1].clone() + errs[0].0[4];
    let mut errs_0_g1 = errs[0].0[..2].to_vec();
    let mut errs_0_g2 = errs[0].1[..2].to_vec();

    new_acc_holder.acc_errs[0].0 = new_acc_holder.acc_errs[0].0[..err1_g1_len - 3].to_vec();
    new_acc_holder.acc_errs[0].0.append(&mut errs_0_g1);
    new_acc_holder.acc_errs[0].0.push(q_term_g1);
    new_acc_holder.acc_errs[0].0.push(l_term_g1);
    new_acc_holder.acc_errs[0].0.push(c_term_g1);
    new_acc_holder.acc_errs[0].1.append(&mut errs_0_g2);
    acc_to_acc_proof(new_acc_holder)
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, true);
    // acc_corr1 = acc_part_corr1 * mu + acc_flat_A_no_blind * acc_flat_B_r + acc_flat_B_no_blind * acc_flat_A_r + srs.Y1P * acc_flat_A_r * acc_flat_B_r
    acc_holder.acc_g1[7] = acc_holder.acc_g1[14] * acc_holder.mu
      + acc_holder.acc_g1[15] * acc_holder.acc_fr[1]
      + acc_holder.acc_g1[16] * acc_holder.acc_fr[0]
      + srs.Y1P * acc_holder.acc_fr[0] * acc_holder.acc_fr[1];
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_holder.acc_g1[..acc_holder.acc_g1.len() - 3].to_vec();
    acc_holder.acc_fr = vec![];
    let acc_proof = acc_to_acc_proof(acc_holder);

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

    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
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
    prev_acc_holder.acc_g1[..7].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g1[11..14].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..7].serialize_uncompressed(&mut bytes).unwrap();
    proof0_11_14.serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    proof.0.iter().enumerate().for_each(|(i, x)| {
      if i < 7 {
        // i==7 is corr1
        let z = *x * acc_gamma + prev_acc_holder.acc_g1[i];
        result &= acc_holder.acc_g1[i] == z;
      }
    });
    proof0_11_14.iter().enumerate().for_each(|(i, x)| {
      let z = *x * acc_gamma + prev_acc_holder.acc_g1[i + 11];
      result &= acc_holder.acc_g1[i + 11] == z;
    });
    result &= acc_holder.acc_g2[0] == prev_acc_holder.acc_g2[0] + proof.1[0] * acc_gamma;
    result &= acc_holder.acc_g2[1] == prev_acc_holder.acc_g2[1] && proof.1[1] == beta_pow_g2 && beta_pow_g2 == acc_holder.acc_g2[1];
    result &= acc_holder.mu == prev_acc_holder.mu + acc_gamma;
    acc_holder.errs[0].0[acc_holder.errs[0].0.len() - 3..]
      .iter()
      .zip(prev_acc_holder.acc_errs[0].0[prev_acc_holder.acc_errs[0].0.len() - 3..].iter())
      .enumerate()
      .for_each(|(j, (x, y))| {
        let z = *y + *x * acc_gamma;
        result &= z == acc_holder.acc_errs[0].0[acc_holder.acc_errs[0].0.len() - 3 + j];
      });
    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);
    let [acc_left_x, acc_left_Q_x, acc_left_zero, acc_left_zero_div, acc_right_x, acc_right_Q_x, acc_right_zero_div, acc_corr1, acc_corr2, acc_corr3, acc_corr4, acc_flat_A, acc_flat_B, acc_flat_C] =
      acc_holder.acc_g1[..]
    else {
      panic!("Wrong acc proof format")
    };
    let [acc_flat_B_g2, beta_pow_g2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    let acc_mu = acc_holder.mu;
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], (srs.X2A[self.m] - srs.X2A[0]).into()));
    temp.push((err_1.0[err_1.1.len() + 1], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 2], srs.Y2A));

    let err_1 = temp;
    let mut acc_1: PairingCheck = vec![
      (acc_flat_A, acc_flat_B_g2),
      ((-acc_left_x * acc_mu).into(), srs.X2A[0]),
      ((-acc_left_Q_x * acc_mu).into(), (srs.X2A[self.m] - srs.X2A[0]).into()),
      (-acc_corr1, srs.Y2A),
    ];
    acc_1.extend(err_1);

    let acc_2: PairingCheck = vec![(acc_flat_B, srs.X2A[0]), (srs.X1A[0], -acc_flat_B_g2)];

    let acc_3: PairingCheck = vec![
      ((acc_left_x - acc_left_zero).into(), srs.X2A[0]),
      (-acc_left_zero_div, srs.X2A[1]),
      (-acc_corr2, srs.Y2A),
    ];

    let acc_4: PairingCheck = vec![
      (acc_flat_C, beta_pow_g2),
      (-acc_right_x, srs.X2A[0]),
      (-acc_right_Q_x, (srs.X2A[self.n] - srs.X2A[0]).into()),
      (-acc_corr3, srs.Y2A),
    ];

    let acc_right_zero: G1Projective = acc_left_zero * (Fr::from(self.m as u32) * Fr::from(self.n as u32).inverse().unwrap());
    let acc_5 = vec![
      ((-acc_right_zero + acc_right_x).into(), srs.X2A[0]),
      (-acc_right_zero_div, srs.X2A[1]),
      (-acc_corr4, srs.Y2A),
    ];

    vec![acc_1, acc_2, acc_3, acc_4, acc_5]
  }

  fn acc_clean_errs(&self, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, false);
    acc_holder.errs = vec![];
    acc_to_acc_proof(acc_holder)
  }
}
