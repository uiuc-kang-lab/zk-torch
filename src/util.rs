use ark_poly::univariate::DensePolynomial;
use ark_std::ops::Mul;
use ark_ec::pairing::Pairing;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain, DenseUVPolynomial, MultilinearExtension, DenseMultilinearExtension};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bn254};
use ark_ff::Field;
use ark_std::{One, Zero, UniformRand, rc::Rc};
use ark_std::rand::RngCore;
use ark_ec::{ScalarMul, VariableBaseMSM};
use ndarray::{Array, IxDyn, s};
use rayon::prelude::*;

use crate::basic_block::Tensor;

pub const SCALE_FACTOR: f64 = (1<<9) as f64;
pub const NV_MAX: usize = 7;

fn bitreverse(mut n: u32, l: u64) -> u32 {
  let mut r = 0;
  for _ in 0..l {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  r
}
pub fn fft_helper<G: ScalarMul + std::ops::MulAssign<Fr>>(a: &mut Vec<G>, domain: GeneralEvaluationDomain<Fr>, inv: bool) {
  let n = a.len();
  let log_size = domain.log_size_of_group();

  let swap = &mut Vec::new();
  (0..n).into_par_iter().map(|i| a[bitreverse(i as u32, log_size) as usize]).collect_into_vec(swap);

  let mut curr = (swap, a);

  let mut m = 1;
  for _ in 0..log_size {
    (0..n)
      .into_par_iter()
      .map(|i| {
        let left = i % (2 * m) < m;
        let k = i / (2 * m) * (2 * m);
        let j = i % m;
        let w = match inv {
          false => domain.element(n / (2 * m) * j),
          true => domain.element(n - n / (2 * m) * j),
        };
        let mut t = curr.0[(k + m) + j];
        t *= w;
        if left {
          return curr.0[k + j] + t;
        } else {
          return curr.0[k + j] - t;
        }
      })
      .collect_into_vec(curr.1);
    curr = (curr.1, curr.0);
    m *= 2;
  }
  if log_size % 2 == 0 {
    (0..n).into_par_iter().map(|i| curr.0[i]).collect_into_vec(curr.1);
  }
}
pub fn fft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, false);
  r
}
pub fn ifft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, true);
  r.par_iter_mut().for_each(|x| *x *= domain.size_inv());
  r
}
pub fn fft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, false);
}
pub fn ifft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, true);
  a.par_iter_mut().for_each(|x| *x *= domain.size_inv());
}

pub fn circulant_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, c: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let lambda = domain.fft(c);
  let mut r = fft(domain, a);
  r.par_iter_mut().enumerate().for_each(|(i, x)| *x *= lambda[i]);
  ifft_in_place(domain, &mut r);
  r
}

pub fn toeplitz_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, m: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let n = (m.len() + 1) / 2;
  let mut temp = m.to_vec();
  let mut m2 = temp.split_off(n - 1);
  m2.push(Fr::zero());
  m2.append(&mut temp);
  let mut temp2 = a.to_vec();
  temp2.resize(2 * n, G::zero());
  let mut r = circulant_mul(domain, &m2, &temp2);
  r.resize(n, G::zero());
  r
}

pub fn msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
  let max_threads = rayon::current_num_threads();
  let size = ark_std::cmp::min(a.len(), b.len());
  if max_threads > size {
    return VariableBaseMSM::msm_unchecked(&a, &b);
  }
  let chunk_size = size / max_threads;
  let a = &a[..size];
  let b = &b[..size];
  let a_chunks = a.par_chunks(chunk_size);
  let b_chunks = b.par_chunks(chunk_size);
  return a_chunks.zip(b_chunks).map(|(x, y)| -> P { VariableBaseMSM::msm_unchecked(&x, &y) }).sum();
}

/// pow helper function
fn integer_to_u64_limbs(exp: u128) -> Vec<u64> {
  let mut limbs: Vec<u64> = Vec::new();
  let mut remaining_exp = exp;

  while remaining_exp > 0 {
    let limb = remaining_exp & 0xFFFFFFFFFFFFFFFF; // Mask to obtain the lower 64 bits
    limbs.push(limb as u64);
    remaining_exp >>= 64; // Right shift by 64 bits
  }

  // If the exponent is zero, make sure to include at least one limb representing zero.
  if limbs.is_empty() {
    limbs.push(0);
  }

  limbs
}

/// univariatization
/// convert a multilinear polynomial to a univariate polynomial
/// by evaluating the polynomial at 2^num_vars points
/// mapping f to U(f)
pub fn univariazation(
  polynomial: &impl MultilinearExtension<Fr>,
) -> DensePolynomial<Fr> {
  let evals = polynomial.to_evaluations();
  DensePolynomial::from_coefficients_vec(evals)
}

/// compute univariatized multilinear quotients U(q_k), k = 0, ..., log_N-1
/// commiter function
pub fn compute_multilinear_quotients(polynomial: &impl MultilinearExtension<Fr>, u_challenge: &[Fr]) -> Vec<DensePolynomial<Fr>> {
  let log_N = u_challenge.len();
  let evaluations = polynomial.to_evaluations();
  // Define the vector of quotients q_k, k = 0, ..., log_N-1
  let mut quotients: Vec<DensePolynomial::<Fr>> = Vec::with_capacity(log_N);
  // Compute the coefficients of q_{n-1}
  let mut size_q = 1 << (log_N - 1);
  let mut q_coeff: Vec<Fr> = Vec::with_capacity(size_q);
  for l in 0..size_q {
    q_coeff.push(evaluations[size_q + l] - evaluations[l]);
  }
  let q: DensePolynomial<Fr> = DensePolynomial::from_coefficients_vec(q_coeff.clone());
  
  quotients.push(q); // [log_N - 1]
  let mut g: Vec<Fr> = evaluations[0..size_q].to_vec();
  // Compute q_k in reverse order from k= n-2, i.e. q_{n-2}, ..., q_0
  for k in 1..log_N {
    // Compute f_k
    let mut f_k: Vec<Fr> = Vec::with_capacity(size_q);
    for l in 0..size_q {
        f_k.push(g[l] + u_challenge[log_N - k] * q_coeff[l]);
    }
    size_q = size_q / 2;
    let mut new_q_coeff: Vec<Fr> = Vec::with_capacity(size_q);
    for l in 0..size_q {
        new_q_coeff.push(f_k[size_q + l] - f_k[l]);
    }
    quotients.push(DensePolynomial::from_coefficients_vec(new_q_coeff.clone())); // [log_N - k - 1]
    q_coeff = new_q_coeff;
    g = f_k.clone();
  }
  quotients.into_iter().rev().collect()
}

/// Compute batched lifted degree quotient polynomial \hat{q}
/// commiter function
pub fn compute_batched_lifted_degree_quotient(quotients: &Vec<DensePolynomial<Fr>>, y_challenge: Fr, N: usize) -> DensePolynomial<Fr> {
  // Batched lifted degree quotient polynomial
  let mut result_coeff: Vec<Fr> = vec![Fr::zero(); N];
  // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
  let mut k = 0;
  let mut scalar = Fr::one(); // y^k
  for quotient in quotients.iter() {
    // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k - 1})
    // then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
    let deg_k = (1 << k) - 1;
    let offset = N - deg_k - 1;
    for idx in 0..(deg_k + 1) {
        let q_coeff = quotient.coeffs();
        if q_coeff.len() > idx {
          result_coeff[offset + idx] += scalar * q_coeff[idx];
        }
    }
    scalar *= y_challenge; // update batching scalar y^k
    k += 1;
  }
  DensePolynomial::from_coefficients_vec(result_coeff.clone())
}

/// compute partially evaluated degree check polynomial \zeta_x
/// commiter function
pub fn compute_partially_evaluated_degree_check_polynomial(batched_quotient: &DensePolynomial<Fr>, quotients: &Vec<DensePolynomial<Fr>>, y_challenge: Fr, x_challenge: Fr) -> DensePolynomial<Fr> {
  let N = batched_quotient.coeffs().len();
  let log_N = quotients.len();
  // Initialize partially evaluated degree check polynomial \zeta_x to \hat{q}
  let mut result = batched_quotient.clone();
  let mut y_power = Fr::one(); // y^k
  for k in 0..log_N {
    // Accumulate y^k * x^{N - d_k - 1} * q_k into \hat{q}
    let deg_k = (1 << k) - 1;
    let challenge_pow= integer_to_u64_limbs((N - deg_k - 1) as u128);
    let x_power = x_challenge.pow(challenge_pow); // x^{N - d_k - 1}
    let x_y_power = -x_power * y_power;
    result = result + quotients[k].mul(x_y_power);
    y_power *= y_challenge; // update batching scalar y^k
  }
  result
}

/// compute z_x
/// commiter function
pub fn compute_partially_evaluated_zeromorph_identity_polynomial(
  f: &DensePolynomial<Fr>,
  quotients: &Vec<DensePolynomial<Fr>>,
  v_evaluation: Fr,
  u_challenge: &[Fr],
  x_challenge: Fr,
) -> DensePolynomial<Fr> {
  let log_N = quotients.len();
  let N = 1<<log_N;
  // Initialize Z_x with f
  let mut result = f.clone();
  // Compute Z_x -= v * \Phi_n(x)
  let N_exp = integer_to_u64_limbs(N);
  let phi_numerator = x_challenge.pow(N_exp) - Fr::one(); // x^N - 1
  let phi_n_x = phi_numerator / (x_challenge - Fr::one());
  result = result + DensePolynomial::from_coefficients_vec(vec![-v_evaluation * phi_n_x]);
  // Add contribution from q_k polynomials
  let mut x_power = x_challenge; // x^{2^k}
  for k in 0..log_N {
    let x_challenge_exp = integer_to_u64_limbs(1 << k);
    let x_1_challenge_exp = integer_to_u64_limbs(1 << (k+1));
    x_power = x_challenge.pow(x_challenge_exp.clone()); // x^{2^k}
    // \Phi_{n-k-1}(x^{2^{k + 1}})
    let phi_term_1 = phi_numerator / (x_challenge.pow(x_1_challenge_exp) - Fr::one());
    // \Phi_{n-k}(x^{2^k})
    let phi_term_2 = phi_numerator / (x_challenge.pow(x_challenge_exp) - Fr::one());
    // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k *  \Phi_{n-k}(x^{2^k})
    let scalar = x_power * phi_term_1 - u_challenge[k] * phi_term_2;
    result = result + quotients[k].mul(-scalar);
  }
  result
}

/// Compute the proof pi
/// commiter function
pub fn compute_batched_evaluation_and_degree_check_quotient(
  zeta_x: DensePolynomial<Fr>,
  Z_x: DensePolynomial<Fr>,
  x_challenge: Fr,
  z_challenge: Fr,
  N_max: usize,
) -> DensePolynomial<Fr> {
  // We cannot commit to polynomials with size > N_max
  let N = zeta_x.coeffs().len();
  assert!(N <= N_max);

  // Compute q_{\zeta} and q_Z in place
  let divisor = DensePolynomial::from_coefficients_vec(vec![-x_challenge, Fr::one()]);
  let q_zeta = &zeta_x / &divisor;
  let q_Z = &Z_x / &divisor;

  // Compute batched quotient q_{\zeta} + z*q_Z
  let batched_quotient = q_zeta.clone() + q_Z.clone().mul(z_challenge);

  // TODO: To complete the degree check, we need to commit to (q_{\zeta} + z*q_Z)*X^{N_max - N - 1}.
  // Verification then requires a pairing check similar to the standard KZG check but with [1]_2 replaced by
  // [X^{N_max - N - 1}]_2. Two issues: A) we do not have an SRS with these G2 elements (so need to generate a fake
  // setup until we can do the real thing), and B) it's not clear how to update our pairing algorithms to do this
  // type of pairing. For now, simply construct q_{\zeta} + z*q_Z without the shift and do a standard KZG
  // pairing check. When we're ready, all we have to do to make this fully legit is commit to the shift here and
  // update the pairing check accordingly. Note: When this is implemented properly, it doesn't make sense to store
  // the (massive) shifted polynomial of size N_max. Ideally, we would only store the unshifted version and just
  // compute the shifted commitment directly via a new method.
  // let batched_shifted_quotient = batched_quotient.clone()*X^{N_max - N - 1}???;
  let mut shift_vec = vec![Fr::zero(); N_max - N - 1];
  shift_vec.append(&mut batched_quotient.coeffs().to_vec());
    
  //let batched_shifted_quotient = batched_quotient.clone();
  let batched_shifted_quotient = DensePolynomial::from_coefficients_vec(shift_vec);
  batched_shifted_quotient
}

/// Compute Commitment of Zeta_x
/// verifier function
pub fn compute_c_zeta_x(
  c_q: G1Affine,
  c_q_k: &Vec<G1Affine>,
  y_challenge: Fr,
  x_challenge: Fr,
) -> G1Affine {
  let log_N = c_q_k.len();
  let N = 1 << log_N; // Check this later
  // Contribution from C_q
  let mut result = c_q.mul(Fr::one());
    
  // Contribution from C_q_k, k = 0, ..., log_n
  for k in 0..log_N {
    let deg_k = (1 << k) - 1;
    // Compute scalar y^k * x^{N - deg_k - 1}
    let y_exp = integer_to_u64_limbs(k as u128);
    let x_exp = integer_to_u64_limbs(N - deg_k - 1);
    let scalar = -y_challenge.pow(y_exp) * x_challenge.pow(x_exp);
    result += c_q_k[k].mul(scalar);
  }
  result.into()
}
  
/// Compute Commitment of Z_x
/// verifier function
pub fn compute_c_z_x(
  f_commitment: G1Affine,
  one_commitment: G1Affine, // G1
  c_q_k: &Vec<G1Affine>,
  v_evaluation: Fr,
  x_challenge: Fr,
  u_challenge: &[Fr],
) -> G1Affine {
  let log_N = c_q_k.len();
  let N = 1 << log_N;
  let mut result = f_commitment.mul(Fr::one());
  // Phi_n(x) = (x^N - 1) / (x - 1)
  let exp_N = integer_to_u64_limbs(N);
  let phi_numerator = x_challenge.pow(exp_N) - Fr::one(); // x^N - 1
  let phi_n_x = phi_numerator / (x_challenge - Fr::one());
  // For now, workaround solution by minus C_v,x outside
  // // Add contribution: -v * \Phi_n(x) * [1]_1
  result += one_commitment.mul(-v_evaluation * phi_n_x);
  // Add contributions: scalar * [q_k],  k = 0,...,log_N, where
  // scalar = -"x" * (x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k * \Phi_{n-k}(x^{2^k}))
  let mut x_pow_2k = x_challenge; // x^{2^k}
  let mut x_pow_2kp1 = x_challenge * x_challenge; // x^{2^{k + 1}}
  for k in 0..log_N {
    let phi_term_1 = phi_numerator / (x_pow_2kp1 - Fr::one()); // \Phi_{n-k-1}(x^{2^{k + 1}})
    let phi_term_2 = phi_numerator / (x_pow_2k - Fr::one()); // \Phi_{n-k}(x^{2^k})
    let scalar = x_pow_2k * phi_term_1 - u_challenge[k] * phi_term_2;
    result += c_q_k[k].mul(-scalar);
    // Update powers of challenge x
    x_pow_2k = x_pow_2kp1;
    x_pow_2kp1 *= x_pow_2kp1;
  }
  result.into()
}

/// Compute Commitment of Zeta_x + z * Commitment of Z_x
/// verifier function
pub fn compute_c_batched(
  c_zeta_x: G1Affine,
  c_z_x: G1Affine,
  z_challenge: Fr,
) -> G1Affine {
  let commitment = c_zeta_x + c_z_x.mul(z_challenge);
  commitment.into()
}

pub struct ZM_proof {
  pub point: Vec<Fr>,
  pub value: Fr,
  pub c_pi: G1Affine,
  pub c_q_k: Vec<G1Affine>,
  pub c_q: G1Affine
}

// P (C,u = (u0, . . . , un−1), v, f, r) → V(C,u, v)
// E2E ZeroMorph
pub fn zm_prove<R: RngCore>(
  srs: (&Vec<G1Affine>,&Vec<G2Affine>),
  poly: &impl MultilinearExtension<Fr>,
  point: &[Fr], // u_challenge
  value: Fr, // v_evaluation
  rng: &mut R,
  nv_max: usize,
) -> ZM_proof {
  let nv = poly.num_vars();
  assert_ne!(nv, 0);

  let uni_poly = univariazation(poly);

  // Committer Step1
  let quotients = compute_multilinear_quotients(poly, &point);

  // Committer Step2
  let y_challenge = Fr::rand(rng);
  let batched_quotient = compute_batched_lifted_degree_quotient(&quotients, y_challenge, 1 << nv);

  // Committer Step3
  let x_challenge = Fr::rand(rng);
  let z_challenge = Fr::rand(rng);
  let zeta_x = compute_partially_evaluated_degree_check_polynomial(&batched_quotient, &quotients, y_challenge, x_challenge);
  let z_x = compute_partially_evaluated_zeromorph_identity_polynomial(&uni_poly, &quotients, value, &point, x_challenge);
  let pi = compute_batched_evaluation_and_degree_check_quotient(zeta_x, z_x, x_challenge, z_challenge, 1 << nv_max);

  // Verifier Step0
  // commit f
  // let uni_poly_coeff = &uni_poly.coeffs;
  // let com = G1Projective::msm(&srs.0[..uni_poly_coeff.len()], uni_poly_coeff).unwrap().into();

  // Verifier Step1
  let mut c_q_k = Vec::new();
  for quotient in quotients.iter() {
    let quotient_coeff = &quotient.coeffs;
    let q_com = G1Projective::msm(&srs.0[..quotient_coeff.len()], quotient_coeff).unwrap().into();
    c_q_k.push(q_com);
  }

  // Verifier Step2
  // commit q_hat
  let batched_quotient_coeff = &batched_quotient.coeffs;
  let q_com = G1Projective::msm(&srs.0[..batched_quotient_coeff.len()], batched_quotient_coeff).unwrap().into();

  // commit pi
  let pi_coeff = &pi.coeffs;
  let pi_com: G1Affine = G1Projective::msm(&srs.0[..pi_coeff.len()], pi_coeff).unwrap().into();

  ZM_proof {
    point: point.to_vec(),
    value: value,
    c_pi: pi_com,
    c_q_k: c_q_k,
    c_q: q_com
  }
}

pub fn zm_verify<R: RngCore>(
  srs: (&Vec<G1Affine>,&Vec<G2Affine>),
  nv_max: usize,
  nv: usize,
  com: G1Affine,
  zm_proof: &ZM_proof,
  rng: &mut R,
) {
  let point = &zm_proof.point;
  let value = zm_proof.value;
  let y_challenge = Fr::rand(rng);
  let x_challenge = Fr::rand(rng);
  let z_challenge = Fr::rand(rng);

  let one_commitment_g1: G1Affine = srs.0[0].clone(); // this way is more efficient than the above one
  
  let zeta_x_com = compute_c_zeta_x(zm_proof.c_q, &zm_proof.c_q_k, y_challenge, x_challenge);
  
  let z_x_com = compute_c_z_x(com, one_commitment_g1, &zm_proof.c_q_k, value, x_challenge, &point);
  
  let c_batched = compute_c_batched(zeta_x_com, z_x_com, z_challenge);
  // pairing
  let x_poly = DensePolynomial::from_coefficients_vec(vec![Fr::zero(), Fr::one()]);
  let mut shift_vec = vec![Fr::zero(); (1<<nv_max) - (1<<nv) - 1];
  shift_vec.append(&mut vec![Fr::one()]);
  let degree_poly = DensePolynomial::from_coefficients_vec(shift_vec);

  let x_poly_coeff = &x_poly.coeffs;
  let x_commitment_g2: G2Affine = G2Projective::msm(&srs.1[..x_poly_coeff.len()], x_poly_coeff).unwrap().into();
  let one_commitment_g2 = srs.1[0].clone();
  let degree_poly_coeff = &degree_poly.coeffs;
  let degree_commitment_g2: G2Affine = G2Projective::msm(&srs.1[..degree_poly_coeff.len()], degree_poly_coeff).unwrap().into();

  let e1 = Bn254::pairing(zm_proof.c_pi, x_commitment_g2+one_commitment_g2.mul(-x_challenge));
  let e2 = Bn254::pairing(c_batched, degree_commitment_g2);
  assert_eq!(e1, e2);
}

// P (C,u = (u0, . . . , un−1), v, f, r) → V(C,u, v)
// E2E ZeroMorph
pub fn test_zm_polynomial<R: RngCore>(
  srs: (&Vec<G1Affine>,&Vec<G2Affine>),
  poly: &impl MultilinearExtension<Fr>,
  point: &[Fr], // u_challenge
  value: Fr, // v_evaluation
  rng: &mut R,
  nv_max: usize,
) {
  let nv = poly.num_vars();
  assert_ne!(nv, 0);
  let uni_poly = univariazation(poly);

  // commit f
  let uni_poly_coeff = &uni_poly.coeffs;
  let com = G1Projective::msm(&srs.0[..uni_poly_coeff.len()], uni_poly_coeff).unwrap().into();

  // ZeroMorph Pairing check
  let zm_proof = zm_prove(srs, poly, point, value, rng, nv_max);

  // pairing
  zm_verify(srs, nv_max, nv, com, &zm_proof, rng);
}
