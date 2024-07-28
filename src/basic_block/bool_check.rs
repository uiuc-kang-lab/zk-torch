use super::BasicBlock;
use crate::{
  basic_block::{Data, DataEnc, SRS},
  onnx, util, PairingCheck, ProveVerifyCache,
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

    // blinding factor for t(x)
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..2).map(|_| Fr::rand(&mut rng2)).collect();

    // compute t(x) = f(x) * (1 - f(x)) / (x^N - 1)
    let one_poly = DensePolynomial { coeffs: vec![Fr::one()] };
    let t_poly = inputs[0].first().unwrap().poly.mul(&one_poly.sub(&inputs[0].first().unwrap().poly));
    let t_poly = t_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![t_x + srs.Y1P * r[0]];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    proof_g1.append(&mut proof_1);
    util::add_randomness(rng, bytes);

    // open f and t at zeta, so that the verifier can check that t(zeta) * zeta^N - 1 = f(zeta) * (1 - f(zeta))
    let zeta = Fr::rand(rng);
    let f_poly_zeta = inputs[0].first().unwrap().poly.evaluate(&zeta);
    let t_poly_zeta = t_poly.evaluate(&zeta);
    let proof_eval = vec![f_poly_zeta, t_poly_zeta];

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof_eval.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // compute h(x) = {f(x) - f(zeta) + gamma * [t(x) - t(zeta)]} / (x - zeta) for proving openings
    let gamma = Fr::rand(rng);
    let gamma_poly = DensePolynomial { coeffs: vec![gamma] };
    let f_zeta_poly = DensePolynomial { coeffs: vec![f_poly_zeta] };
    let t_zeta_poly = DensePolynomial { coeffs: vec![t_poly_zeta] };
    let temp1 = &inputs[0].first().unwrap().poly.sub(&f_zeta_poly);
    let temp2 = &t_poly.sub(&t_zeta_poly);
    let x_minus_zeta = DensePolynomial {
      coeffs: vec![-zeta, Fr::one()],
    };
    let h_poly = &temp1.add(&temp2.mul(&gamma_poly)) / &x_minus_zeta;
    let h_poly_x = util::msm::<G1Projective>(&srs.X1A, &h_poly.coeffs) + srs.Y1P * r[1];
    proof_g1.push(h_poly_x);

    // blinding factor
    let C = srs.X1P[0] * (inputs[0].first().unwrap().r + gamma * r[0] + zeta * r[1]) - srs.X1P[1] * r[1];
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
    let [t_x, h_x, C] = proof.0[..] else {
      panic!("proof.0 should have 3 elements")
    };
    let proof0_for_check = &proof.0[..1];
    let mut bytes = Vec::new();
    proof0_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let zeta = Fr::rand(rng);

    let [f_zeta, t_zeta] = proof.2[..] else {
      panic!("proof.2 should have 2 elements")
    };
    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_poly_z = vanishing_poly.evaluate(&zeta);
    // verifier first checks that f(zeta) * (1 - f(zeta)) = t(zeta) * vanishing_poly(zeta)
    assert!(f_zeta * (Fr::one() - f_zeta) == t_zeta * vanishing_poly_z);
    // verifier then checks openings are correct by pairing e(., .):
    // e([h(x)]_1, [x-z]_2) = e([f(x)]_1 - [f(z)]_1 - gamma * ([t(x)]_1 + [t(z)]_1), [1]_2)
    // which is equivalent to
    // e([h(x)]_1, [x]_2) = e([f(x)]_1 - [f(z)]_1 - gamma * ([t(x)]_1 + [t(z)]_1) + zeta * [h(x)]_1, [1]_2)
    // note: the above equation does not include the blinding factor C
    let proof_for_check = &proof.2[..2];
    let mut bytes = Vec::new();
    proof_for_check.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let gamma = Fr::rand(rng);
    let mut check_for_opening_at_z = inputs[0].first().unwrap().g1 + t_x * gamma;
    check_for_opening_at_z -= srs.X1A[0] * (f_zeta + gamma * t_zeta);
    check_for_opening_at_z += h_x * zeta;

    let checks = vec![(h_x, srs.X2A[1]), ((-check_for_opening_at_z).into(), srs.X2A[0]), (C.into(), srs.Y2A)];
    vec![checks]
  }
}
