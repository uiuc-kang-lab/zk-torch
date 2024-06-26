use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_ec::Group;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{ops::{Div, Mul}, ops::Sub, UniformRand, One, Zero};
use ndarray::{azip, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[0].shape() == inputs[1].shape());
    let mut r = ArrayD::zeros(inputs[0].dim());
    azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x * y);
    vec![r]
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
    let inp0 = &inputs[0].first().unwrap();
    let inp1 = &inputs[1].first().unwrap();
    let out = &outputs[0].first().unwrap();
    let N = inp0.raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    
    // compute quotient commitment
    let q = inp0.poly.mul(&inp1.poly).sub(&out.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let qx = util::msm::<G1Projective>(&srs.X1A, &q.coeffs);

    // add randomness
    let mut quotient_bytes = Vec::new();
    qx.serialize_compressed(&mut quotient_bytes).unwrap();
    util::add_randomness(rng, quotient_bytes);

    // compute openings
    let lambda = Fr::rand(rng);
    let polys = vec![inp0.poly.clone(), inp1.poly.clone(), out.poly.clone(), q];
    let openings: Vec<Fr> = polys.iter().map(|p| p.evaluate(&lambda)).collect();

    // compute opening commitment
    let opening_rand = Fr::rand(rng);
    let mut curr = Fr::one();
    let mut opening_poly = DensePolynomial::zero();
    polys.iter().enumerate().for_each(|(i, p)| {
      opening_poly += &(p - &DensePolynomial::from_coefficients_vec(vec![openings[i]])).mul(curr);
      curr *= opening_rand;
    });

    opening_poly = opening_poly.div(&DensePolynomial::from_coefficients_vec(vec![-lambda, Fr::one()]));

    let opening_com = util::msm::<G1Projective>(&srs.X1A, &opening_poly.coeffs);
    
    return (vec![qx, opening_com], Vec::new(), openings);
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
    let mut checks = vec![];

    let input_commitments = vec![inputs[0].first().unwrap().g1, inputs[1].first().unwrap().g1];
    let output_commitment = outputs[0].first().unwrap().g1;
    let qx = proof.0[0];
    let opening_com = proof.0[1];
    let openings = proof.2.clone();

    // add randomness
    let mut quotient_bytes = Vec::new();
    qx.serialize_compressed(&mut quotient_bytes).unwrap();
    util::add_randomness(rng, quotient_bytes);

    // check opening
    let lambda = Fr::rand(rng);
    let total = openings[0] * openings[1] - openings[2];
    assert!(total == openings[3] * (lambda.pow(&[inputs[0].first().unwrap().len as u64]) - Fr::one()));

    // check opening commitment
    let opening_rand = Fr::rand(rng);

    let mut pows = vec![Fr::one()];
    for _ in 0..3 {
      pows.push(pows.last().unwrap() * &opening_rand);
    }

    let coms: Vec<_> = vec![input_commitments[0], input_commitments[1], output_commitment, qx];
    let mut opening_check = util::msm::<G1Projective>(&coms, &pows);
    opening_check -= G1Projective::generator().mul(openings.iter().zip(pows.iter()).map(|(a, b)| a * b).sum::<Fr>());

    checks.push(vec![
      (opening_com, srs.X2A[1]),
      ((-opening_com.mul(lambda) - opening_check).into(), srs.X2A[0]),
    ]);
    checks
  }
}
