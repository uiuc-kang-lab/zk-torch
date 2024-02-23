use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use ndarray::{azip, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> ArrayD<Fr> {
    let mut r = ArrayD::zeros(inputs[0].shape());
    azip!((&x in inputs[0], &y in inputs[1], z in &mut r) *z = x * y);
    r
  }
  fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    output: &Data,
    _rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = inputs[0].raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = util::msm::<G2Projective>(&srs.1[..N], &inputs[1].poly.coeffs) + srs.1[srs.1.len() - 1] * inputs[1].r;
    let gx2 = gx2.into();
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = util::msm::<G1Projective>(&srs.0[..N - 1], &t.coeffs);

    // Blinding
    let mut rng = StdRng::from_entropy();
    let r = Fr::rand(&mut rng);
    let tx = (tx + srs.0[srs.1.len() - 1] * r).into();
    let C = (inputs[0].g1 * inputs[1].r) + (inputs[1].g1 * inputs[0].r) + (srs.0[srs.1.len() - 1] * (inputs[0].r * inputs[1].r))
      - (srs.0[0] * output.r)
      - ((srs.0[N] - srs.0[0]) * r);
    let C = C.into();
    return (vec![tx, C], vec![gx2]);
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bn254::pairing(inputs[0].g1, proof.1[0]) - Bn254::pairing(output.g1, srs.1[0]);
    let rhs = Bn254::pairing(proof.0[0], srs.1[inputs[0].len] - srs.1[0]) + Bn254::pairing(proof.0[1], srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
    // Verify gx2
    let lhs = Bn254::pairing(inputs[1].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], proof.1[0]);
    assert!(lhs == rhs);
  }
}
