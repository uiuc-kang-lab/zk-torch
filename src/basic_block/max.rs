use super::BasicBlock;
use crate::{
  basic_block::{Data, DataEnc, SRS},
  onnx,
  util::{self, calc_pow, convert_to_data},
  PairingCheck, ProveVerifyCache,
};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::One;
use ark_std::Zero;
use ark_std::{cmp::max, UniformRand};
use ndarray::Axis;
use ndarray::{arr0, arr1, azip, Array, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use std::{
  borrow::Borrow,
  ops::{Mul, Sub},
  time::Instant,
};

#[derive(Debug)]
pub struct MaxBasicBlock;
impl BasicBlock for MaxBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    vec![arr1(&[inputs[0].fold(Fr::zero(), |max, x| {
      if *x < Fr::from(1 << 28) && *x > max {
        return *x;
      } else {
        return max;
      }
    })])
    .into_dyn()]
  }
}

#[derive(Debug)]
pub struct MaxProofBasicBlock;

// This max includes a proof and is intended to be followed by a lookup range check on the second output.
impl BasicBlock for MaxProofBasicBlock {
  // Returns the max of the input and max - x for all x in input
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    let cq_max = Fr::from(-onnx::CQ_RANGE_LOWER);
    let max_arr = inputs[0]
      .fold_axis(Axis(0), Fr::from(onnx::CQ_RANGE_LOWER), |max, y| {
        if (*y < cq_max && *y > *max) || (*max > cq_max && *y < cq_max) || (*y > cq_max && *max > cq_max && *y > *max) {
          *y
        } else {
          *max
        }
      })
      .into_shape(vec![1])
      .unwrap();
    let max_val = max_arr.first().unwrap();

    let mut r = ArrayD::zeros(inputs[0].shape());
    azip!((r in &mut r, &x in inputs[0]) *r = *max_val - x);
    vec![max_arr, r]
  }

  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let N = outputs[1].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Round 1: Prove difference and commit s
    // Diff proving
    let max = outputs[0].first().unwrap();
    let input = inputs[0].first().unwrap();
    let diff = outputs[1].first().unwrap();
    let C1 = srs.X1P[0] * (max.r - input.r - diff.r);

    // s has to have 0 as the first element, which should happen after sorting
    let mut s = diff.raw.clone();
    s.sort();
    let s_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&s));
    let s_x = util::msm::<G1Projective>(&srs.X1A, &s_poly.coeffs);

    // Round 2: Commit Z
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..2).map(|_| Fr::rand(&mut rng2)).collect();
    let mut proof = vec![s_x + srs.Y1P * r[0], C1];
    proof.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // Compute Z commitment
    // Grand product argument to check that s is a permutation of f
    let mut Z = vec![Fr::zero(); N];
    let gamma = Fr::rand(rng);
    Z[0] = Fr::one();
    for j in 1..N {
      Z[j] = Z[j - 1] * (gamma + diff.raw[j - 1]) * (gamma + s[j - 1]).inverse().unwrap();
    }
    let Z_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&Z));
    let Z_blind: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    let Z_blind_poly = DensePolynomial::from_coefficients_vec(vec![Z_blind[0], Z_blind[1], Z_blind[2]]);
    let Z_poly = &Z_poly + &Z_blind_poly.mul(&DensePolynomial::from(domain.vanishing_polynomial()));
    let Z_x = util::msm::<G1Projective>(&srs.X1A, &Z_poly.coeffs);

    // Compute Z(omega * X) polynomial
    let Zg_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };

    // Calculate L0(X)(Z(X)-1) polynomial
    let mut L0 = vec![Fr::zero(); N];
    L0[0] = Fr::one();
    let L0_poly = DensePolynomial { coeffs: domain.ifft(&L0) };
    let one = DensePolynomial { coeffs: vec![Fr::one()] };
    let L0Z_poly = L0_poly.mul(&Z_poly.sub(&one));

    // Calculate L0(X)s(X) polynomial
    let L0s_poly = L0_poly.mul(&s_poly);

    // Round 3: Commit t
    // Fiat-Shamir
    let mut bytes = Vec::new();
    vec![Z_x].serialize_uncompressed(&mut bytes).unwrap();
    proof.push(Z_x);
    util::add_randomness(rng, bytes);

    let alpha = Fr::rand(rng);

    // t constraints:
    // Z(oX) (s(X) + gamma) = (d(X) + gamma) * Z(X)
    // L0(X)(Z(X)-1) = 0s
    // L0(X)s(X) = 0
    let gamma_poly = DensePolynomial::from_coefficients_vec(vec![gamma]);
    let alpha_poly = DensePolynomial::from_coefficients_vec(vec![alpha]);
    let t_poly = &Zg_poly.mul(&(&s_poly + &gamma_poly)) - &Z_poly.mul(&(&diff.poly + &gamma_poly))
      + L0Z_poly.mul(&alpha_poly)
      + L0s_poly.mul(&alpha_poly).mul(&alpha_poly);
    let t_poly = t_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Round 4: Compute openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![t_x + srs.Y1P * r[1]];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    proof.append(&mut proof_1);
    util::add_randomness(rng, bytes);

    let zeta = Fr::rand(rng);
    let omega = domain.group_gen();
    let L0_z = L0_poly.evaluate(&(zeta));
    let Z_z = Z_poly.evaluate(&(zeta));
    let Z_gz = Z_poly.evaluate(&(omega * zeta));
    let evals = vec![Z_z, Z_gz, L0_z];

    // Round 5: Commit opening proofs
    // Fiat-Shamir
    let mut bytes = Vec::new();
    evals.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);

    // Calculate linearization polynomial
    let zeta_pows = calc_pow(zeta, N);
    let Z_z_poly = DensePolynomial::from_coefficients_vec(vec![Z_z]);
    let Z_gz_poly = DensePolynomial::from_coefficients_vec(vec![Z_gz]);
    let r_poly = (&Z_gz_poly.mul(&(&s_poly + &gamma_poly)) - &(Z_z_poly.mul(&(&diff.poly + &gamma_poly)))
      + Z_poly.sub(&one).mul(&DensePolynomial::from_coefficients_vec(vec![alpha * L0_z]))
      + s_poly.mul(&DensePolynomial::from_coefficients_vec(vec![alpha * alpha * L0_z])))
    .sub(&DensePolynomial::from_coefficients_vec(vec![zeta_pows[N - 1] - Fr::one()]).mul(&t_poly));

    // Compute opening argument for r and Z over zeta
    let W_V = DensePolynomial {
      coeffs: vec![-zeta, Fr::one()],
    };
    let W_Q = &(&r_poly + &(&Z_poly - &Z_z_poly).mul(&DensePolynomial::from_coefficients_vec(vec![v]))) / &W_V;
    let W_x = util::msm::<G1Projective>(&srs.X1A, &W_Q.coeffs);

    // Compute opening argument for Z over omega * zeta
    let W_gV = DensePolynomial {
      coeffs: vec![-zeta * omega, Fr::one()],
    };
    let W_gQ: DensePolynomial<_> = &Z_poly.sub(&Z_gz_poly) / &W_gV;
    let W_gx = util::msm::<G1Projective>(&srs.X1A, &W_gQ.coeffs);

    // Round 5 end randomness
    let mut bytes = Vec::new();
    vec![W_x, W_gx].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _ = Fr::rand(rng);
    proof.append(&mut vec![W_x, W_gx]);
    proof.push(srs.X1P[0] * (r[1] * (zeta_pows[N - 1] - Fr::one()) + diff.r * Z_z - (Z_gz + alpha * alpha * L0_z) * r[0]));

    (proof, Vec::new(), evals)
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = outputs[1].first().unwrap().len;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let max = outputs[0].first().unwrap();
    let input = inputs[0].first().unwrap();
    let diff = outputs[1].first().unwrap();

    let [s_x, C1, Z_x, t_x, W_x, W_gx, C2] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [Z_z, Z_gz, L0_z] = proof.2[..] else { panic!("Wrong proof format") };

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![s_x, C1].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let gamma = Fr::rand(rng);

    // Round 3 randomness
    let mut bytes = Vec::new();
    vec![Z_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let alpha = Fr::rand(rng);

    // Round 4 randomness
    let mut bytes = Vec::new();
    vec![t_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let zeta = Fr::rand(rng);

    // Round 5 randomness
    let mut bytes = Vec::new();
    proof.2.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);

    // Round 5 end randomness
    let mut bytes = Vec::new();
    vec![W_x, W_gx].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let u = Fr::rand(rng);

    // Verify that diff = max - input
    checks.push(vec![((max.g1 - input.g1 - diff.g1).into(), srs.X2A[0]), (-C1, srs.Y2A)]);

    // Verify t batched check
    let zeta_pows = calc_pow(zeta, N);
    let omega = domain.group_gen();
    let r_0 = Z_gz * gamma - Z_z * gamma - L0_z * alpha;
    let D = s_x * (Z_gz + L0_z * alpha * alpha) - diff.g1 * Z_z + Z_x * (L0_z * alpha + u + v) - t_x * (zeta_pows[N - 1] - Fr::one());
    let E = srs.X1P[0] * (-r_0 + u * Z_gz + v * Z_z);
    checks.push(vec![
      ((W_x + W_gx * u).into(), srs.X2A[1]),
      ((-(W_x * zeta + W_gx * u * omega * zeta + D - E)).into(), srs.X2A[0]),
      (-C2, srs.Y2A),
    ]);
    checks
  }
}
