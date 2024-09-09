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

// OneToOneBasicBlock is a basic block that checks if there exists a one-to-one mapping between
// data and sorted_data. Besides, it checks the indices are sorted in the same way as the data.
#[derive(Debug)]
pub struct OneToOneBasicBlock;
impl BasicBlock for OneToOneBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 4);
    assert!(inputs[0].ndim() == 1 && inputs[1].ndim() == 1 && inputs[2].ndim() == 1 && inputs[3].ndim() == 1);
    assert!(inputs[0].len() == inputs[1].len() && inputs[0].len() == inputs[2].len() && inputs[0].len() == inputs[3].len());
    vec![]
  }

  // The high-level proving idea:
  // there exists a one-to-one mapping between f(x) and f_s(x), i.e., f_s(sigma(omega^i))=f(omega^i) for all i in [0, N). sigma is a permutation.
  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let N = inputs[0].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // prover blinding factors
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..17).map(|_| Fr::rand(&mut rng2)).collect();

    // Round 1: sort data and indices, and then commit all data carriers
    let data = inputs[0].first().unwrap().raw.clone();
    let indices = inputs[1].first().unwrap().raw.clone();
    let sorted_data = inputs[2].first().unwrap().raw.clone();
    let sorted_indices = inputs[3].first().unwrap().raw.clone();

    // The reason we don't batch these g polys is that we need to use them separately for Z poly proving,
    // it may be more time consuming to batch them here.
    // compute g(x) = f(x) + r_f(x) * (x^N - 1) for secure opening
    // r_f(x) is a polynomial of degree 2
    let r_f_poly = DensePolynomial {
      coeffs: vec![r[0], r[1], r[2]],
    };
    let g_poly = inputs[0].first().unwrap().poly.clone().add(r_f_poly.mul_by_vanishing_poly(domain));
    let g_x = util::msm::<G1Projective>(&srs.X1A, &g_poly.coeffs);

    // compute g_idx(x) = f_idx(x) + r_f_idx(x) * (x^N - 1) for secure opening
    let r_f_idx_poly = DensePolynomial {
      coeffs: vec![r[3], r[4], r[5]],
    };
    let g_idx_poly = inputs[1].first().unwrap().poly.clone().add(r_f_idx_poly.mul_by_vanishing_poly(domain));
    let g_idx_x = util::msm::<G1Projective>(&srs.X1A, &g_idx_poly.coeffs);

    // compute g_s(x) = f_s(x) + r_f_s(x) * (x^N - 1) for secure opening
    let r_f_s_poly = DensePolynomial {
      coeffs: vec![r[6], r[7], r[8]],
    };
    let g_s_poly = inputs[2].first().unwrap().poly.clone().add(r_f_s_poly.mul_by_vanishing_poly(domain));
    let g_s_x = util::msm::<G1Projective>(&srs.X1A, &g_s_poly.coeffs);

    // compute g_idx_s(x) = f_idx_s(x) + r_f_idx_s(x) * (x^N - 1) for secure opening
    let r_f_idx_s_poly = DensePolynomial {
      coeffs: vec![r[9], r[10], r[11]],
    };
    let g_idx_s_poly = inputs[3].first().unwrap().poly.clone().add(r_f_idx_s_poly.mul_by_vanishing_poly(domain));
    let g_idx_s_x = util::msm::<G1Projective>(&srs.X1A, &g_idx_s_poly.coeffs);

    let minus_one_poly = DensePolynomial::from_coefficients_vec(vec![-Fr::one()]);
    // compute q(x) = [f(x) - g(x)]/ (x^N - 1) for proving f(x) = g(x)
    let q_poly = r_f_poly.mul(&minus_one_poly);
    let r_Q = r[12];
    let q_x = util::msm::<G1Projective>(&srs.X1A, &q_poly.coeffs) + srs.Y1P * r_Q;
    let r_plus_r_Q = DensePolynomial { coeffs: vec![r_Q] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![inputs[0].first().unwrap().r],
    });
    let r_plus_r_Q_x = util::msm::<G1Projective>(&srs.X1A, &r_plus_r_Q.coeffs);

    // compute q_idx(x) = [f_idx(x) - g_idx(x)]/ (x^N - 1) for proving f_idx(x) = g_idx(x)
    let q_idx_poly = r_f_idx_poly.mul(&minus_one_poly);
    let r_Q_idx = r[13];
    let q_idx_x = util::msm::<G1Projective>(&srs.X1A, &q_idx_poly.coeffs) + srs.Y1P * r_Q_idx;
    let r_idx_plus_r_Q_idx = DensePolynomial { coeffs: vec![r_Q_idx] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![inputs[1].first().unwrap().r],
    });
    let r_idx_plus_r_Q_idx_x = util::msm::<G1Projective>(&srs.X1A, &r_idx_plus_r_Q_idx.coeffs);

    // compute q_s(x) = [f_s(x) - g_s(x)]/ (x^N - 1) for proving f_s(x) = g_s(x)
    let q_s_poly = r_f_s_poly.mul(&minus_one_poly);
    let r_Q_s = r[14];
    let q_s_x = util::msm::<G1Projective>(&srs.X1A, &q_s_poly.coeffs) + srs.Y1P * r_Q_s;
    let r_s_plus_r_Q_s = DensePolynomial { coeffs: vec![r_Q_s] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![inputs[2].first().unwrap().r],
    });
    let r_s_plus_r_Q_s_x = util::msm::<G1Projective>(&srs.X1A, &r_s_plus_r_Q_s.coeffs);

    // compute q_idx_s(x) = [f_idx_s(x) - g_idx_s(x)]/ (x^N - 1) for proving f_idx_s(x) = g_idx_s(x)
    let q_idx_s_poly = r_f_idx_s_poly.mul(&minus_one_poly);
    let r_Q_idx_s = r[15];
    let q_idx_s_x = util::msm::<G1Projective>(&srs.X1A, &q_idx_s_poly.coeffs) + srs.Y1P * r_Q_idx_s;
    let r_idx_s_plus_r_Q_idx_s = DensePolynomial { coeffs: vec![r_Q_idx_s] }.mul_by_vanishing_poly(domain).sub(&DensePolynomial {
      coeffs: vec![inputs[3].first().unwrap().r],
    });
    let r_idx_s_plus_r_Q_idx_s_x = util::msm::<G1Projective>(&srs.X1A, &r_idx_s_plus_r_Q_idx_s.coeffs);

    // Round 2: Commit Z
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut rng2 = StdRng::from_entropy();
    let mut proof = vec![g_x, g_idx_x, g_s_x, g_idx_s_x, q_x, q_idx_x, q_s_x, q_idx_s_x];
    proof.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // Compute Z commitment
    // Grand product argument to check that g_s and g_idx_s is a permutation of g and g_idx
    // Z(omega * x) * (g_s(x) + beta * g_idx_s(x) + gamma) = Z(X) * (g(x) + beta * g_idx(x) + gamma)
    let mut Z = vec![Fr::zero(); N];
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);
    Z[0] = Fr::one();
    for j in 1..N {
      Z[j] =
        Z[j - 1] * (gamma + beta * indices[j - 1] + data[j - 1]) * (gamma + beta * sorted_indices[j - 1] + sorted_data[j - 1]).inverse().unwrap();
    }
    let Z_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&Z));
    let Z_blind: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    let Z_blind_poly = DensePolynomial::from_coefficients_vec(vec![Z_blind[0], Z_blind[1], Z_blind[2]]);
    let Z_poly = &Z_poly + &Z_blind_poly.mul(&DensePolynomial::from(domain.vanishing_polynomial()));
    let Z_x = util::msm::<G1Projective>(&srs.X1A, &Z_poly.coeffs);

    // Compute Z(omega * x) polynomial
    let Z_omega_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };

    // Calculate L0Z(x) = L0(x)(Z(x)-1) polynomial
    let mut L0 = vec![Fr::zero(); N];
    L0[0] = Fr::one();
    let L0_poly = DensePolynomial { coeffs: domain.ifft(&L0) };
    let one = DensePolynomial { coeffs: vec![Fr::one()] };
    let L0Z_poly = L0_poly.mul(&Z_poly.sub(&one));

    // Round 3: Commit t
    // Fiat-Shamir
    let mut bytes = Vec::new();
    vec![Z_x].serialize_uncompressed(&mut bytes).unwrap();
    proof.push(Z_x);
    util::add_randomness(rng, bytes);

    let alpha = Fr::rand(rng);

    // t constraints:
    // The following eqs should hold for all x in the domain_N:
    // - Z(omega * x) * (g_s(x) + beta * g_idx_s(x) + gamma) = Z(X) * (g(x) + beta * g_idx(x) + gamma)
    // - L0(x)(Z(x)-1) = 0
    let gamma_poly = DensePolynomial::from_coefficients_vec(vec![gamma]);
    let beta_poly = DensePolynomial::from_coefficients_vec(vec![beta]);
    let alpha_poly = DensePolynomial::from_coefficients_vec(vec![alpha]);
    let t_N_poly = &Z_omega_poly.mul(&(&g_s_poly + &beta_poly.mul(&g_idx_s_poly).add(gamma_poly.clone())))
      - &Z_poly.mul(&(&g_poly + &beta_poly.mul(&g_idx_poly).add(gamma_poly)))
      + L0Z_poly.mul(&alpha_poly);
    let t_poly = t_N_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Round 4: Compute openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![t_x + srs.Y1P * r[16]];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    proof.append(&mut proof_1);
    util::add_randomness(rng, bytes);

    let zeta = Fr::rand(rng);
    let omega = domain.group_gen();
    let t_zeta = t_poly.evaluate(&(zeta));
    let Z_zeta = Z_poly.evaluate(&(zeta));
    let Z_omega_zeta = Z_poly.evaluate(&(omega * zeta));
    let g_zeta = g_poly.evaluate(&(zeta));
    let g_idx_zeta = g_idx_poly.evaluate(&(zeta));
    let g_s_zeta = g_s_poly.evaluate(&(zeta));
    let g_idx_s_zeta = g_idx_s_poly.evaluate(&(zeta));

    let evals = vec![
      t_zeta,       // t(zeta)
      Z_zeta,       // Z(zeta)
      Z_omega_zeta, // Z(omega * zeta)
      g_zeta,       // g(zeta)
      g_idx_zeta,   // g_idx(zeta)
      g_s_zeta,     // g_s(zeta)
      g_idx_s_zeta, // g_idx_s(zeta)
    ];

    let evals_polys = evals.iter().map(|&x| DensePolynomial { coeffs: vec![x] }).collect::<Vec<_>>();

    // Round 5: Commit opening proofs
    // Fiat-Shamir
    let mut bytes = Vec::new();
    evals.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);
    let v_pows = calc_pow(v, 6);

    // compute h(x) = SUM_i{[poly_i(x) - poly_i(zeta)] * v^i} / (x - zeta) for proving openings at zeta are correct
    let h_denominator = DensePolynomial::from_coefficients_vec(vec![-zeta, Fr::one()]);
    let h_numerator_terms = vec![
      t_poly.clone().sub(&evals_polys[0].clone()),
      Z_poly.clone().sub(&evals_polys[1].clone()),
      g_poly.clone().sub(&evals_polys[3].clone()),
      g_idx_poly.clone().sub(&evals_polys[4].clone()),
      g_s_poly.clone().sub(&evals_polys[5].clone()),
      g_idx_s_poly.clone().sub(&evals_polys[6].clone()),
    ];
    let h_numerator = h_numerator_terms.iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, x)| {
      acc.add(x.mul(&DensePolynomial::from_coefficients_vec(vec![v_pows[i]])))
    });
    let h_poly = &h_numerator / &h_denominator;
    let h_x = util::msm::<G1Projective>(&srs.X1A, &h_poly.coeffs);

    // compute h'(x) = SUM_i{[poly_i(x) - poly_i(omega * zeta)] * v^i} / (x - omega * zeta) for proving openings at omega * zeta are correct
    let h_prime_denominator = DensePolynomial::from_coefficients_vec(vec![-omega * zeta, Fr::one()]);
    let h_prime_numerator_terms = vec![Z_poly.clone().sub(&evals_polys[2].clone())];
    let h_prime_numerator = h_prime_numerator_terms.iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, x)| {
      acc.add(x.mul(&DensePolynomial::from_coefficients_vec(vec![v_pows[i]])))
    });
    let h_prime_poly = &h_prime_numerator / &h_prime_denominator;
    let h_prime_x = util::msm::<G1Projective>(&srs.X1A, &h_prime_poly.coeffs);

    // Round 5 end randomness.
    let mut bytes = Vec::new();
    let mut proof1 = vec![h_x, h_prime_x];
    proof1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _u = Fr::rand(rng);
    let w = Fr::rand(rng);
    let w_pows = calc_pow(w, 3);
    proof.append(&mut proof1);

    // blinding
    let mut C = r_plus_r_Q_x + r_idx_plus_r_Q_idx_x * w_pows[0] + r_s_plus_r_Q_s_x * w_pows[1] + r_idx_s_plus_r_Q_idx_s_x * w_pows[2];
    C += srs.X1P[0] * r[16] * v_pows[0];
    proof.push(C);

    (proof, Vec::new(), evals)
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = inputs[1].first().unwrap().len;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let data = inputs[0].first().unwrap().g1;
    let indices = inputs[1].first().unwrap().g1;
    let sorted_data = inputs[2].first().unwrap().g1;
    let sorted_indices = inputs[3].first().unwrap().g1;

    let [g_x, g_idx_x, g_s_x, g_idx_s_x, q_x, q_idx_x, q_s_x, q_idx_s_x, Z_x, t_x, h_x, h_prime_x, C] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [
      t_zeta,         // t(zeta)
      Z_zeta,         // Z(zeta)
      Z_omega_zeta,   // Z(omega * zeta)
      g_zeta,         // g(zeta)
      g_idx_zeta,     // g_idx(zeta)
      g_s_zeta,       // g_s(zeta)
      g_idx_s_zeta,   // g_idx_s(zeta)
    ] = proof.2[..] else { panic!("Wrong eval proof format") };

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![g_x, g_idx_x, g_s_x, g_idx_s_x, q_x, q_idx_x, q_s_x, q_idx_s_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let beta = Fr::rand(rng);
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
    vec![h_x, h_prime_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let u = Fr::rand(rng);
    let w = Fr::rand(rng);
    let w_pows = calc_pow(w, 3);

    // Verify that t(zeta) * (zeta^N - 1) =
    //   Z(omega * zeta) * (g_s(zeta) + beta * g_idx_s(zeta) + gamma) - Z(zeta) * (g(zeta) + beta * g_idx(zeta) + gamma)
    //   + L0(zeta)(Z(zeta)-1) * alpha
    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_poly_zeta = vanishing_poly.evaluate(&zeta);
    let mut L0 = vec![Fr::zero(); N];
    L0[0] = Fr::one();
    let L0_poly = DensePolynomial { coeffs: domain.ifft(&L0) };
    let L0_zeta = L0_poly.evaluate(&zeta);

    let A_zeta = Z_omega_zeta * (g_s_zeta + beta * g_idx_s_zeta + gamma) - Z_zeta * (g_zeta + beta * g_idx_zeta + gamma)
      + L0_zeta * (Z_zeta - Fr::one()) * alpha;
    assert!(t_zeta * vanishing_poly_zeta == A_zeta);

    // Verify openings are correct
    let v_pows = calc_pow(v, 6);
    let mut check_for_opening_at_zeta =
      t_x * v_pows[0] + Z_x * v_pows[1] + g_x * v_pows[2] + g_idx_x * v_pows[3] + g_s_x * v_pows[4] + g_idx_s_x * v_pows[5];
    check_for_opening_at_zeta -= srs.X1A[0]
      * (t_zeta * v_pows[0] + Z_zeta * v_pows[1] + g_zeta * v_pows[2] + g_idx_zeta * v_pows[3] + g_s_zeta * v_pows[4] + g_idx_s_zeta * v_pows[5]);
    let mut check_for_opening_at_omega_zeta = Z_x * v_pows[0];
    check_for_opening_at_omega_zeta -= srs.X1A[0] * (Z_omega_zeta * v_pows[0]);
    let check_for_g_eq_f =
      (data - g_x) + (indices - g_idx_x) * w_pows[0] + (sorted_data - g_s_x) * w_pows[1] + (sorted_indices - g_idx_s_x) * w_pows[2];
    let check_for_g_eq_f_q_terms = q_x + q_idx_x * w_pows[0] + q_s_x * w_pows[1] + q_idx_s_x * w_pows[2];

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
