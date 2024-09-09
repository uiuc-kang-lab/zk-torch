use super::BasicBlock;
use crate::{
  basic_block::{Data, DataEnc, PairingCheck, ProveVerifyCache, SRS},
  onnx,
  util::{self, calc_pow},
};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::{cmp::max, One, UniformRand, Zero};
use ndarray::{arr0, arr1, azip, s, ArrayD, Axis};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

// OrderedBasicBlock is a basic block that computes
// d(omega^i) = f_s(x) - f_s(omega * x) for all i in [0, N)
// which can be used to prove that the data (shape: (N,)) is ordered by checking
// all d(omega^i) for i in [0, N-1) are non-negative if descending, or non-positive if ascending.
// It takes one input: the sorted data tensor f_s.
// It returns one tensor: the differences d.
#[derive(Debug)]
pub struct OrderedBasicBlock;
impl BasicBlock for OrderedBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);

    let mut sorted_data_shift_1 = inputs[0].into_iter().skip(1).cloned().collect::<Vec<_>>();
    sorted_data_shift_1.push(*inputs[0].first().unwrap());
    let sorted_data_shift_1 = arr1(&sorted_data_shift_1);

    // Outsource the range check of d to the caller
    let diff = (inputs[0] - sorted_data_shift_1).into_dyn();
    vec![diff]
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let N = inputs[0].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let mut proof = vec![];

    // prover blinding factors
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();

    // Round 1: sort data and indices, and then commit all data carriers
    let data = inputs[0].first().unwrap();
    let diff = outputs[0].first().unwrap();

    // compute g_s(x) = f_s(x) + r_f_s(x) * (x^N - 1) for secure opening
    let r_f_s_poly = DensePolynomial {
      coeffs: vec![r[0], r[1], r[2]],
    };
    let g_s_poly = data.poly.clone().add(r_f_s_poly.mul_by_vanishing_poly(domain));
    let g_s_x = util::msm::<G1Projective>(&srs.X1A, &g_s_poly.coeffs);

    // compute g_d(x) = d(x) + r_d(x) * (x^N - 1) for secure opening
    let r_diff_poly = DensePolynomial {
      coeffs: vec![r[3], r[4], r[5]],
    };
    let g_diff_poly = diff.poly.clone().add(r_diff_poly.mul_by_vanishing_poly(domain));
    let g_diff_x = util::msm::<G1Projective>(&srs.X1A, &g_diff_poly.coeffs);

    let minus_one_poly = DensePolynomial::from_coefficients_vec(vec![-Fr::one()]);
    // compute q_s(x) = [f_s(x) - g_s(x)]/ (x^N - 1) for proving f_s(x) = g_s(x)
    let q_s_poly = r_f_s_poly.mul(&minus_one_poly);
    let r_Q_s = r[6];
    let q_s_x = util::msm::<G1Projective>(&srs.X1A, &q_s_poly.coeffs) + srs.Y1P * r_Q_s;
    let r_s_plus_r_Q_s = DensePolynomial { coeffs: vec![r_Q_s] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![inputs[0].first().unwrap().r],
    });
    let r_s_plus_r_Q_s_x = util::msm::<G1Projective>(&srs.X1A, &r_s_plus_r_Q_s.coeffs);

    // compute q_diff(x) = [d(x) - g_diff(x)]/ (x^N - 1) for proving d(x) = g_diff(x)
    let q_diff_poly = r_diff_poly.mul(&minus_one_poly);
    let r_Q_diff = r[7];
    let q_diff_x = util::msm::<G1Projective>(&srs.X1A, &q_diff_poly.coeffs) + srs.Y1P * r_Q_diff;
    let r_diff_plus_r_Q_diff = DensePolynomial { coeffs: vec![r_Q_diff] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![outputs[0].first().unwrap().r],
    });
    let r_diff_plus_r_Q_diff_x = util::msm::<G1Projective>(&srs.X1A, &r_diff_plus_r_Q_diff.coeffs);

    // Compute g_s(omega * x) polynomial
    let g_s_omega_poly = DensePolynomial {
      coeffs: g_s_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };

    // Commit t

    // t constraints:
    // The following eqs should hold for all x in the domain_N:
    // - g_d(x) + g_s(omega * x) - g_s(x) = 0
    let t_N_poly = &g_diff_poly.clone().add(g_s_omega_poly.clone()).sub(&g_s_poly);
    let t_poly = t_N_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Round 2: Compute openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![g_s_x, g_diff_x, q_s_x, q_diff_x, t_x + srs.Y1P * r[8]];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    proof.append(&mut proof_1);
    util::add_randomness(rng, bytes);

    let zeta = Fr::rand(rng);
    let omega = domain.group_gen();
    let t_zeta = t_poly.evaluate(&(zeta));
    let g_s_zeta = g_s_poly.evaluate(&(zeta));
    let g_s_omega_zeta = g_s_omega_poly.evaluate(&(zeta));
    let g_diff_zeta = g_diff_poly.clone().evaluate(&(zeta));

    let evals = vec![
      t_zeta,         // t(zeta)
      g_s_zeta,       // g_s(zeta)
      g_s_omega_zeta, // g_s(omega * zeta)
      g_diff_zeta,    // g_diff(zeta)
    ];

    let evals_polys = evals.iter().map(|&x| DensePolynomial { coeffs: vec![x] }).collect::<Vec<_>>();

    // Round 3: Commit opening proofs
    // Fiat-Shamir
    let mut bytes = Vec::new();
    evals.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);
    let v_pows = calc_pow(v, 3);

    // compute h(x) = SUM_i{[poly_i(x) - poly_i(zeta)] * v^i} / (x - zeta) for proving openings at zeta are correct
    let h_denominator = DensePolynomial::from_coefficients_vec(vec![-zeta, Fr::one()]);
    let h_numerator_terms = vec![
      t_poly.clone().sub(&evals_polys[0].clone()),
      g_s_poly.clone().sub(&evals_polys[1].clone()),
      g_diff_poly.clone().sub(&evals_polys[3].clone()),
    ];
    let h_numerator = h_numerator_terms.iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, x)| {
      acc.add(x.mul(&DensePolynomial::from_coefficients_vec(vec![v_pows[i]])))
    });
    let h_poly = &h_numerator / &h_denominator;
    let h_x = util::msm::<G1Projective>(&srs.X1A, &h_poly.coeffs);

    // compute h'(x) = SUM_i{[poly_i(x) - poly_i(omega * zeta)] * v^i} / (x - omega * zeta) for proving openings at omega * zeta are correct
    let h_prime_denominator = DensePolynomial::from_coefficients_vec(vec![-omega * zeta, Fr::one()]);
    let h_prime_numerator_terms = vec![g_s_poly.clone().sub(&evals_polys[2].clone())];
    let h_prime_numerator = h_prime_numerator_terms.iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, x)| {
      acc.add(x.mul(&DensePolynomial::from_coefficients_vec(vec![v_pows[i]])))
    });
    let h_prime_poly = &h_prime_numerator / &h_prime_denominator;
    let h_prime_x = util::msm::<G1Projective>(&srs.X1A, &h_prime_poly.coeffs);

    // Round 3 end randomness.
    let mut bytes = Vec::new();
    let mut proof1 = vec![h_x, h_prime_x];
    proof1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _u = Fr::rand(rng);
    let w = Fr::rand(rng);
    let w_pows = calc_pow(w, 2);
    proof.append(&mut proof1);

    // blinding
    let mut C = r_s_plus_r_Q_s_x * w_pows[0] + r_diff_plus_r_Q_diff_x * w_pows[1];
    C += srs.X1P[0] * r[8] * v_pows[0];
    proof.push(C);

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
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = inputs[0].first().unwrap().len;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let data = inputs[0].first().unwrap().g1;
    let diff = outputs[0].first().unwrap().g1;

    let [g_s_x, g_diff_x, q_s_x, q_diff_x, t_x, h_x, h_prime_x, C] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [
      t_zeta,         // t(zeta)
      g_s_zeta,       // g_s(zeta)
      g_s_omega_zeta, // g_s(omega * zeta)
      g_diff_zeta,    // g_diff(zeta)
    ] = proof.2[..] else { panic!("Wrong eval proof format") };

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![g_s_x, g_diff_x, q_s_x, q_diff_x, t_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let zeta = Fr::rand(rng);

    // Round 3 randomness
    let mut bytes = Vec::new();
    proof.2.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);

    // Round 3 end randomness
    let mut bytes = Vec::new();
    vec![h_x, h_prime_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let u = Fr::rand(rng);
    let w = Fr::rand(rng);
    let w_pows = calc_pow(w, 2);

    // Verify that t(zeta) * (zeta^N - 1) = d(zeta) + g_s(omega * zeta) - g_s(zeta)
    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_poly_zeta = vanishing_poly.evaluate(&zeta);
    assert!(t_zeta * vanishing_poly_zeta == g_diff_zeta + g_s_omega_zeta - g_s_zeta);

    // Verify openings are correct
    let v_pows = calc_pow(v, 3);
    let mut check_for_opening_at_zeta = t_x * v_pows[0] + g_s_x * v_pows[1] + g_diff_x * v_pows[2];
    check_for_opening_at_zeta -= srs.X1A[0] * (t_zeta * v_pows[0] + g_s_zeta * v_pows[1] + g_diff_zeta * v_pows[2]);
    let mut check_for_opening_at_omega_zeta = g_s_x * v_pows[0];
    check_for_opening_at_omega_zeta -= srs.X1A[0] * (g_s_omega_zeta * v_pows[0]);
    let check_for_g_eq_f = (data - g_s_x) * w_pows[0] + (diff - g_diff_x) * w_pows[1];
    let check_for_g_eq_f_q_terms = q_s_x * w_pows[0] + q_diff_x * w_pows[1];

    checks.push(vec![
      ((h_x + h_prime_x * u).into(), srs.X2A[1]),
      (
        (check_for_g_eq_f - (check_for_opening_at_zeta + check_for_opening_at_omega_zeta * u + h_x * zeta + h_prime_x * (u * omega * zeta))).into(),
        srs.X2A[0],
      ),
      ((-check_for_g_eq_f_q_terms).into(), (srs.X2A[N] - srs.X2A[0]).into()),
      (C.into(), srs.Y2A),
    ]);

    checks
  }
}
