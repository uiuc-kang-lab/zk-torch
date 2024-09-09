use super::BasicBlock;
use crate::{
  basic_block::{Data, DataEnc, PairingCheck, ProveVerifyCache, SRS},
  onnx, util,
};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::{One, UniformRand, Zero};
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use std::ops::{Add, Mul, Sub};
use tract_onnx::tract_core::num_traits::ops::bytes;

// BooleanCheckBasicBlock is a basic block that checks if all elements in inputs[0] are boolean values (0 or 1).
// The high-level proving idea:
// Given that N elements in inputs are all boolean, we can encode them as a polynomial f(x), where f(omega^i) = 0 or 1 for all i in [0, N).
// Then the bool check is equivalent to proving the existence of t(x), where t(x) = f(x) * (1 - f(x)) / (x^N - 1)
// (It’s because the N roots of x^N-1 are omega^i for all i in [0, N). This implies the existence of polynomial t(x). Otherwise, t(x) will not exist.)
// Note: To ensure secure opening, we replace f(x) with g(x) = f(x) + r_f(x) * (x^N - 1), where r_f(x) is a polynomial of degree 2.
#[derive(Debug)]
pub struct BooleanCheckBasicBlock;
impl BasicBlock for BooleanCheckBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // check if all elements are 0 or 1
    assert!(inputs.len() == 1);
    assert!(inputs[0].iter().all(|y| {
      let y_int = util::fr_to_int(*y);
      y_int == 0 || y_int == 1
    }));
    vec![]
  }

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
    let mut proof_g1 = Vec::new();

    // blinding factors
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..6).map(|_| Fr::rand(&mut rng2)).collect();

    // compute g(x) = f(x) + r_f(x) * (x^N - 1) for secure opening
    // r_f(x) is a polynomial of degree 2
    let r_f_poly = DensePolynomial {
      coeffs: vec![r[0], r[1], r[2]],
    };
    let g_poly = inputs[0].first().unwrap().poly.clone().add(r_f_poly.mul_by_vanishing_poly(domain));
    let g_x = util::msm::<G1Projective>(&srs.X1A, &g_poly.coeffs);

    // compute q(x) = [g(x) - f(x)] / (x^N - 1) for proving g(x) = f(x) over the domain
    let q_poly = g_poly.clone().sub(&inputs[0].first().unwrap().poly.clone()).divide_by_vanishing_poly(domain).unwrap().0;
    let r_Q = r[3];
    let q_x = util::msm::<G1Projective>(&srs.X1A, &q_poly.coeffs) + srs.Y1P * r_Q;

    // compute t(x) = g(x) * (1 - g(x)) / (x^N - 1)
    let one_poly = DensePolynomial { coeffs: vec![Fr::one()] };
    let t_poly = g_poly.clone().mul(&one_poly.sub(&g_poly.clone()));
    let t_poly = t_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![t_x + srs.Y1P * r[4], g_x, q_x];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    proof_g1.append(&mut proof_1);
    util::add_randomness(rng, bytes);

    // open g and t at zeta, so that the verifier can check that t(zeta) * zeta^N - 1 = g(zeta) * (1 - g(zeta))
    let zeta = Fr::rand(rng);
    let g_poly_zeta = g_poly.clone().evaluate(&zeta);
    let t_poly_zeta = t_poly.evaluate(&zeta);
    let proof_eval = vec![g_poly_zeta, t_poly_zeta];

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof_eval.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // compute h(x) = {g(x) - g(zeta) + gamma * [t(x) - t(zeta)]} / (x - zeta) for proving openings are correct
    let gamma = Fr::rand(rng);
    let gamma_poly = DensePolynomial { coeffs: vec![gamma] };
    let g_zeta_poly = DensePolynomial { coeffs: vec![g_poly_zeta] };
    let t_zeta_poly = DensePolynomial { coeffs: vec![t_poly_zeta] };
    let temp1 = &g_poly.clone().sub(&g_zeta_poly);
    let temp2 = &t_poly.sub(&t_zeta_poly);
    let x_minus_zeta = DensePolynomial {
      coeffs: vec![-zeta, Fr::one()],
    };
    let h_poly = &temp1.add(&temp2.mul(&gamma_poly)) / &x_minus_zeta;
    let h_poly_x = util::msm::<G1Projective>(&srs.X1A, &h_poly.coeffs) + srs.Y1P * r[5];
    proof_g1.push(h_poly_x);

    // compute r + r_Q * (x^N - 1) for considering the blinding factor in secure opening
    let r_plus_r_Q = DensePolynomial {
      coeffs: vec![inputs[0].first().unwrap().r],
    }
    .add(DensePolynomial { coeffs: vec![r_Q] }.mul_by_vanishing_poly(domain));
    let r_plus_r_Q_x = util::msm::<G1Projective>(&srs.X1A, &r_plus_r_Q.coeffs);

    // blinding factor
    let C = srs.X1P[0] * (gamma * r[4] + zeta * r[5]) - srs.X1P[1] * r[5] + r_plus_r_Q_x;
    proof_g1.push(C);

    (proof_g1, vec![], proof_eval)
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
    let N = inputs[0].first().unwrap().len;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let [t_x, g_x, q_x, h_x, C] = proof.0[..] else {
      panic!("proof.0 should have 5 elements")
    };
    let proof0_for_check = &proof.0[..3];
    let mut bytes = Vec::new();
    proof0_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let zeta = Fr::rand(rng);

    let [g_zeta, t_zeta] = proof.2[..] else {
      panic!("proof.2 should have 2 elements")
    };
    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_poly_z = vanishing_poly.evaluate(&zeta);
    // verifier first checks that f(zeta) * (1 - f(zeta)) = t(zeta) * vanishing_poly(zeta)
    assert!(g_zeta * (Fr::one() - g_zeta) == t_zeta * vanishing_poly_z);

    // verifier then checks openings are correct by pairing e(., .):
    // e([h(x)]_1, [x-z]_2) = e([g(x)]_1 - [g(z)]_1 + gamma * ([t(x)]_1 - [t(z)]_1), [1]_2)
    // which is equivalent to
    // e([h(x)]_1, [x]_2) = e([g(x)]_1 - [g(z)]_1 + gamma * ([t(x)]_1 - [t(z)]_1) + zeta * [h(x)]_1, [1]_2)
    // note: the above equation does not include the blinding factor C
    let proof_for_check = &proof.2[..2];
    let mut bytes = Vec::new();
    proof_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let gamma = Fr::rand(rng);
    let mut check_for_opening_at_z = g_x + t_x * gamma;
    check_for_opening_at_z -= srs.X1A[0] * (g_zeta + gamma * t_zeta);
    check_for_opening_at_z -= g_x - inputs[0].first().unwrap().g1; // g(x) - f(x)
    check_for_opening_at_z += h_x * zeta;

    let checks = vec![
      (h_x, srs.X2A[1]),
      ((-check_for_opening_at_z).into(), srs.X2A[0]),
      (-q_x, (srs.X2A[inputs[0].first().unwrap().len] - srs.X2A[0]).into()),
      (C.into(), srs.Y2A),
    ];
    vec![checks]
  }
}
