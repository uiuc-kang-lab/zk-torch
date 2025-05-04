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
use ndarray::{Array, ArrayD, Axis};
use rand::{rngs::StdRng, SeedableRng};

define_acc_terms!(
  PermuteG1Terms,
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
    Flat_L,
    Flat_R
  ],
  []
);
define_acc_terms!(PermuteG2Terms, [B_g2, D_g2], []);
define_acc_terms!(PermuteFrTerms, [], []);
define_acc_err_terms!(PermuteErrG1Terms);
define_acc_err_terms!(PermuteErrG2Terms);
define_acc_err_terms!(PermuteErrFrTerms);
define_acc_err_terms!(PermuteErrGtTerms);

impl AccProofLayout for PermuteBasicBlock {
  fn acc_g1_num(&self, _is_prover: bool) -> usize {
    PermuteG1Terms::<G1Projective>::COUNT
  }

  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    PermuteG2Terms::<G2Projective>::COUNT
  }

  fn acc_fr_num(&self, _is_prover: bool) -> usize {
    PermuteFrTerms::<Fr>::COUNT
  }

  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: Vec::new(),
      mu: Fr::one(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    }
  }

  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: proof.1.clone(),
      acc_fr: Vec::new(),
      mu: Fr::one(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    }
  }

  fn mira_prove(
    &self,
    _srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
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
    let acc1 = PermuteG1Terms::<G1Projective>::from_vec(&acc_1.acc_g1);
    let acc2 = PermuteG1Terms::<G1Projective>::from_vec(&acc_2.acc_g1);
    let acc1_fiat_shamir = vec![
      acc1.Left_x,
      acc1.Left_Q_x,
      acc1.Left_zero,
      acc1.Left_zero_div,
      acc1.Right_x,
      acc1.Right_Q_x,
      acc1.Right_zero_div,
      acc1.Flat_L,
      acc1.Flat_R,
    ];
    let acc2_fiat_shamir = vec![
      acc2.Left_x,
      acc2.Left_Q_x,
      acc2.Left_zero,
      acc2.Left_zero_div,
      acc2.Right_x,
      acc2.Right_Q_x,
      acc2.Right_zero_div,
      acc2.Flat_L,
      acc2.Flat_R,
    ];
    acc1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc2_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    new_acc_holder.acc_g1 = acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.acc_g2 = acc_1.acc_g2.clone();
    new_acc_holder.mu = acc_1.mu + acc_gamma * acc_2.mu;
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
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let acc1 = PermuteG1Terms::<G1Affine>::from_vec(&acc_1.acc_g1);
    let acc2 = PermuteG1Terms::<G1Affine>::from_vec(&acc_2.acc_g1);
    let acc1_fiat_shamir = vec![
      acc1.Left_x,
      acc1.Left_Q_x,
      acc1.Left_zero,
      acc1.Left_zero_div,
      acc1.Right_x,
      acc1.Right_Q_x,
      acc1.Right_zero_div,
      acc1.Flat_L,
      acc1.Flat_R,
    ];
    let acc2_fiat_shamir = vec![
      acc2.Left_x,
      acc2.Left_Q_x,
      acc2.Left_zero,
      acc2.Left_zero_div,
      acc2.Right_x,
      acc2.Right_Q_x,
      acc2.Right_zero_div,
      acc2.Flat_L,
      acc2.Flat_R,
    ];
    acc1_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    acc2_fiat_shamir.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    acc_2.acc_g1.iter().enumerate().for_each(|(i, x)| {
      let z = *x * acc_gamma + acc_1.acc_g1[i];
      result &= new_acc.acc_g1[i] == z;
    });
    result &= new_acc.acc_g2[0] == acc_1.acc_g2[0];
    result &= new_acc.acc_g2[1] == acc_1.acc_g2[1];
    result &= new_acc.mu == acc_1.mu + acc_gamma * acc_2.mu;
    Some(result)
  }
}

#[derive(Debug)]
pub struct PermuteBasicBlock {
  pub permutation: (Vec<usize>, Vec<usize>),
  pub n: usize,
  pub m: usize,
}

// Permute elements of a 2d matrix into another 2d matrix
// This is proven via this equation:
// [alpha^0,alpha^1,alpha^2,...] A [alpha^0,alpha^n,alpha^(2n),...]^T
//                                =
// [alpha^(p_0[0]),alpha^(p_0[1]),alpha^(p_0[2]),...] B [alpha^(p_1[0]),alpha^(p_1[1]),alpha^(p_1[2]),...]^T
// Where A is in the input matrix, B is the output matrix, and p is the permutation
// In order to do a matrix transpose, we set p_0[i]=ni and p_1[i]=i
impl BasicBlock for PermuteBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 2);
    let n = inputs[0].len_of(Axis(0));
    Ok(vec![Array::from_shape_fn(
      (self.permutation.0.len(), self.permutation.1.len()),
      |(i, j)| {
        let s = self.permutation.0[i] + self.permutation.1[j];
        assert!(s < inputs[0].len());
        inputs[0][[s % n, s / n]]
      },
    )
    .into_dyn()])
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
    let alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("permute_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
    };
    let (input, output) = (inputs[0], outputs[0]);

    // n rows, m columns in input
    let n = input.len();
    let m = input[0].raw.len();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    // n2 rows, m2 columns in output
    let n2 = self.permutation.0.len();
    let m2 = self.permutation.1.len();
    let domain_m2 = GeneralEvaluationDomain::<Fr>::new(m2).unwrap();

    let alpha_pow = calc_pow(alpha, n * m);

    let b = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::Data(b) = cache.entry(format!("permute_b_msm_{m}_{n}")).or_insert_with(|| {
        CacheValues::Data({
          let b: Vec<_> = (0..m).map(|i| alpha_pow[i * n]).collect();
          Data::new(srs, &b)
        })
      }) else {
        panic!("Cache type error")
      };
      b.clone()
    };

    let d = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::Data(d) = cache.entry(format!("permute_d_msm_{self:p}")).or_insert_with(|| {
        CacheValues::Data({
          let d: Vec<_> = (0..m2).map(|i| alpha_pow[self.permutation.1[i]]).collect();
          Data::new(srs, &d)
        })
      }) else {
        panic!("Cache type error")
      };
      d.clone()
    };

    let mut flat_L = vec![Fr::zero(); m];
    let mut flat_L_r = Fr::zero();
    for i in 0..n {
      for j in 0..m {
        flat_L[j] += input[i].raw[j] * alpha_pow[i];
      }
      flat_L_r += input[i].r * alpha_pow[i];
    }
    let mut flat_L = Data::new(srs, &flat_L);
    flat_L.r = flat_L_r;

    let mut flat_R = vec![Fr::zero(); m2];
    let mut flat_R_r = Fr::zero();
    for i in 0..n2 {
      for j in 0..m2 {
        flat_R[j] += output[i].raw[j] * alpha_pow[self.permutation.0[i]];
      }
      flat_R_r += output[i].r * alpha_pow[self.permutation.0[i]];
    }
    let mut flat_R = Data::new(srs, &flat_R);
    flat_R.r = flat_R_r;

    // Calculate Left
    let left_raw: Vec<Fr> = (0..m).map(|i| flat_L.raw[i] * alpha_pow[i * n]).collect();
    let left_poly = DensePolynomial::from_coefficients_vec(domain_m.ifft(&left_raw));
    let left_x = util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs);
    let left_Q_poly = flat_L.poly.mul(&b.poly).sub(&left_poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let left_Q_x = util::msm::<G1Projective>(&srs.X1A, &left_Q_poly.coeffs);
    let left_zero = srs.X1A[0] * (Fr::from(m as u32).inverse().unwrap() * left_raw.iter().sum::<Fr>());
    let left_zero_div = if left_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &left_poly.coeffs[1..])
    };

    // Calculate Right
    let right_raw: Vec<Fr> = (0..m2).map(|i| flat_R.raw[i] * alpha_pow[self.permutation.1[i]]).collect();
    let right_poly = DensePolynomial::from_coefficients_vec(domain_m2.ifft(&right_raw));
    let right_x = util::msm::<G1Projective>(&srs.X1A, &right_poly.coeffs);
    let right_Q_poly = flat_R.poly.mul(&d.poly).sub(&right_poly).divide_by_vanishing_poly(domain_m2).unwrap().0;
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
    let mut corr = vec![
      -(srs.X1P[m] - srs.X1P[0]) * r[1] + b.g1 * flat_L.r - srs.X1P[0] * r[0],
      -srs.X1P[1] * r[3] + srs.X1P[0] * (r[0] - r[2]),
      -(srs.X1P[m2] - srs.X1P[0]) * r[5] + d.g1 * flat_R.r - srs.X1P[0] * r[4],
      -srs.X1P[1] * r[6] + srs.X1P[0] * (r[4] - r[2] * Fr::from(m as u32) * Fr::from(m2 as u32).inverse().unwrap()),
    ];
    proof.append(&mut corr);
    let mut proof2 = vec![];
    #[cfg(feature = "fold")]
    {
      let b_g2 = {
        let mut cache = cache.lock().unwrap();
        let CacheValues::G2(b_g2) = cache
          .entry(format!("permute_b_msm_g2_{m}_{n}"))
          .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &b.poly.coeffs).into()))
        else {
          panic!("Cache type error")
        };
        b_g2.clone()
      };

      let d_g2 = {
        let mut cache = cache.lock().unwrap();
        let CacheValues::G2(d_g2) = cache
          .entry(format!("permute_d_msm_g2_{self:p}"))
          .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &d.poly.coeffs).into()))
        else {
          panic!("Cache type error")
        };
        d_g2.clone()
      };
      let mut additional_g2_for_acc: Vec<G2Projective> = vec![b_g2.into(), d_g2.into()];
      let mut additional_g1_for_acc = vec![flat_L.g1 + srs.Y1P * flat_L.r, flat_R.g1 + srs.Y1P * flat_R.r];
      proof.append(&mut additional_g1_for_acc);
      proof2.append(&mut additional_g2_for_acc);
    }

    return (proof, proof2, Vec::new());
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
    let alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("permute_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
    };
    let (input, output) = (inputs[0], outputs[0]);

    // n rows, m columns in input
    let n = input.len();
    let m = input[0].len;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    // n2 rows, m2 columns in output
    let n2 = self.permutation.0.len();
    let m2 = self.permutation.1.len();
    let domain_m2 = GeneralEvaluationDomain::<Fr>::new(m2).unwrap();

    let [left_x, left_Q_x, left_zero, left_zero_div, right_x, right_Q_x, right_zero_div, corr1, corr2, corr3, corr4] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let alpha_pow = calc_pow(alpha, n * m);
    let b: Vec<_> = (0..m).map(|i| alpha_pow[i * n]).collect();
    let c: Vec<_> = (0..n2).map(|i| alpha_pow[self.permutation.0[i]]).collect();
    let d: Vec<_> = (0..m2).map(|i| alpha_pow[self.permutation.1[i]]).collect();

    let b_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(b_g2) = cache
        .entry(format!("permute_b_msm_g2_{m}_{n}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_m.ifft(&b)).into()))
      else {
        panic!("Cache type error")
      };
      b_g2.clone()
    };

    let d_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(d_g2) = cache
        .entry(format!("permute_d_msm_g2_{self:p}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_m2.ifft(&d)).into()))
      else {
        panic!("Cache type error")
      };
      d_g2.clone()
    };

    // Calculate flat_L
    let temp: Vec<_> = (0..n).map(|i| input[i].g1).collect();
    let flat_L_g1 = util::msm::<G1Projective>(&temp, &alpha_pow[..n]).into();

    // Calculate flat_R
    let temp: Vec<_> = (0..n2).map(|i| output[i].g1).collect();
    let flat_R_g1 = util::msm::<G1Projective>(&temp, &c).into();

    // Check left(x) (left = flat_L * b)
    checks.push(vec![
      (flat_L_g1, b_g2),
      (-left_x, srs.X2A[0]),
      (-left_Q_x, (srs.X2A[m] - srs.X2A[0]).into()),
      (-corr1, srs.Y2A),
    ]);

    // Check left(x) - left(0) is divisible by x
    checks.push(vec![
      ((left_x - left_zero).into(), srs.X2A[0]),
      (-left_zero_div, srs.X2A[1]),
      (-corr2, srs.Y2A),
    ]);

    // Check right(x) (right = flat_R * d)
    checks.push(vec![
      (flat_R_g1, d_g2),
      (-right_x, srs.X2A[0]),
      (-right_Q_x, (srs.X2A[m2] - srs.X2A[0]).into()),
      (-corr3, srs.Y2A),
    ]);

    // Assume right(0) = left(0)*m/m2 (which assumes ∑left=∑right)
    let right_zero: G1Affine = (left_zero * (Fr::from(m as u32) * Fr::from(m2 as u32).inverse().unwrap())).into();

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
    let _alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("permute_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
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
    _srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: AccProofProjRef,
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), AccProofAffine) {
    let cqlin_proof_g1 = proof.0.to_vec();
    let cqlin_proof_g2 = proof.1.to_vec();
    (
      (
        cqlin_proof_g1.iter().map(|x| (*x).into()).collect(),
        cqlin_proof_g2.iter().map(|x| (*x).into()).collect(),
        vec![],
      ),
      (
        acc_proof.0.iter().map(|x| (*x).into()).collect(),
        acc_proof.1.iter().map(|x| (*x).into()).collect(),
        acc_proof.2.to_vec(),
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

    let alpha = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::RLCRandom(alpha) = cache.entry("permute_alpha".to_owned()).or_insert_with(|| CacheValues::RLCRandom(Fr::rand(rng))) else {
        panic!("Cache type error")
      };
      alpha.clone()
    };
    let (input, output) = (inputs[0], outputs[0]);

    // n rows, m columns in input
    let n = input.len();
    let m = input[0].len;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    // n2 rows, m2 columns in output
    let n2 = self.permutation.0.len();
    let m2 = self.permutation.1.len();
    let domain_m2 = GeneralEvaluationDomain::<Fr>::new(m2).unwrap();

    let alpha_pow = calc_pow(alpha, n * m);
    let b: Vec<_> = (0..m).map(|i| alpha_pow[i * n]).collect();
    let c: Vec<_> = (0..n2).map(|i| alpha_pow[self.permutation.0[i]]).collect();
    let d: Vec<_> = (0..m2).map(|i| alpha_pow[self.permutation.1[i]]).collect();

    let b_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(b_g2) = cache
        .entry(format!("permute_b_msm_g2_{m}_{n}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_m.ifft(&b)).into()))
      else {
        panic!("Cache type error")
      };
      b_g2.clone()
    };

    let d_g2 = {
      let mut cache = cache.lock().unwrap();
      let CacheValues::G2(d_g2) = cache
        .entry(format!("permute_d_msm_g2_{self:p}"))
        .or_insert_with(|| CacheValues::G2(util::msm::<G2Projective>(&srs.X2A, &domain_m2.ifft(&d)).into()))
      else {
        panic!("Cache type error")
      };
      d_g2.clone()
    };

    // Calculate flat_L
    let temp: Vec<_> = (0..n).map(|i| input[i].g1).collect();
    let flat_L: G1Affine = util::msm::<G1Projective>(&temp, &alpha_pow[..n]).into();

    // Calculate flat_R
    let temp: Vec<_> = (0..n2).map(|i| output[i].g1).collect();
    let flat_R: G1Affine = util::msm::<G1Projective>(&temp, &c).into();

    let proof_g1 = PermuteG1Terms::<G1Affine>::from_vec(&proof.0);
    let acc_g2 = PermuteG2Terms::<G2Affine>::from_vec(&acc_proof.1);
    result &= acc_g2.B_g2 == b_g2 && acc_g2.D_g2 == d_g2;
    result &= proof_g1.Flat_L == flat_L && proof_g1.Flat_R == flat_R;
    if prev_acc_proof.2.len() == 0 && acc_proof.2[0].is_one() {
      return Some(result);
    }

    let proof = self.verifier_proof_to_acc(proof);
    let prev_acc_holder = acc_proof_to_holder(self, prev_acc_proof, true);
    let acc_holder = acc_proof_to_holder(self, acc_proof, true);
    result &= self.mira_verify(prev_acc_holder, proof, acc_holder, rng).unwrap();
    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: AccProofAffineRef) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let m2 = self.permutation.1.len();
    let acc_holder = acc_proof_to_holder(self, acc_proof, false);

    let acc_g1 = PermuteG1Terms::<G1Affine>::from_vec(&acc_holder.acc_g1);
    let acc_g2 = PermuteG2Terms::<G2Affine>::from_vec(&acc_holder.acc_g2);

    let acc_1: PairingCheck = vec![
      (acc_g1.Flat_L, acc_g2.B_g2),
      (-acc_g1.Left_x, srs.X2A[0]),
      (-acc_g1.Left_Q_x, (srs.X2A[self.m] - srs.X2A[0]).into()),
      (-acc_g1.Corr1, srs.Y2A),
    ];

    let acc_2: PairingCheck = vec![
      ((acc_g1.Left_x - acc_g1.Left_zero).into(), srs.X2A[0]),
      (-acc_g1.Left_zero_div, srs.X2A[1]),
      (-acc_g1.Corr2, srs.Y2A),
    ];

    let acc_3: PairingCheck = vec![
      (acc_g1.Flat_R, acc_g2.D_g2),
      (-acc_g1.Right_x, srs.X2A[0]),
      (-acc_g1.Right_Q_x, (srs.X2A[m2] - srs.X2A[0]).into()),
      (-acc_g1.Corr3, srs.Y2A),
    ];

    let acc_right_zero: G1Projective = acc_g1.Left_zero * (Fr::from(self.m as u32) * Fr::from(m2 as u32).inverse().unwrap());
    let acc_4 = vec![
      ((-acc_right_zero + acc_g1.Right_x).into(), srs.X2A[0]),
      (-acc_g1.Right_zero_div, srs.X2A[1]),
      (-acc_g1.Corr4, srs.Y2A),
    ];

    let pairing_zero = PairingOutput::<Bn<ark_bn254::Config>>::zero();
    vec![(acc_1, pairing_zero), (acc_2, pairing_zero), (acc_3, pairing_zero), (acc_4, pairing_zero)]
  }
}
