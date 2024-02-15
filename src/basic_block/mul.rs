use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub};
use rand::Rng;

pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock {
  fn run(_model: &Data, inputs: &Vec<Vec<Fr>>) -> Vec<Fr> {
    let mut r = Vec::new();
    for i in 0..inputs[0].len() {
      r.push(inputs[0][i] * inputs[1][i]);
    }
    return r;
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<Data>,
    output: &Data,
    _rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = inputs[0].raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = util::msm::<G2Projective>(&srs.1[..N], &inputs[1].poly.coeffs).into();
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = util::msm::<G1Projective>(&srs.0[..N - 1], &t.coeffs).into();
    return (vec![tx], vec![gx2]);
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut R,
  ) {
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bn254::pairing(inputs[0].g1, proof.1[0]) - Bn254::pairing(output.g1, srs.1[0]);
    let rhs = Bn254::pairing(proof.0[0], srs.1[inputs[0].dims[0]] - srs.1[0]);
    assert!(lhs == rhs);
    // Verify gx2
    let lhs = Bn254::pairing(inputs[1].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], proof.1[0]);
    assert!(lhs == rhs);
  }
}
