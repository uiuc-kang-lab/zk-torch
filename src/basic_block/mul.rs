use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use rand::{rngs::StdRng, SeedableRng};

pub struct MulConstBasicBlock {
  pub c: usize,
}
impl BasicBlock for MulConstBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().map(|x| *x * Fr::from(self.c as u32)).collect()]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let C = srs.X1P[0] * (Fr::from(self.c as u32) * inputs[0].r - outputs[0].r);
    return (vec![C], vec![]);
  }
  fn verify(
    &self,
    srs: &SRS,
    _model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    let lhs = Bn254::pairing(inputs[0].g1, srs.X2P[0] * Fr::from(self.c as u32));
    let rhs = Bn254::pairing(outputs[0].g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
  }
}
pub struct MulScalarBasicBlock;
impl BasicBlock for MulScalarBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().map(|x| *x * inputs[1][0]).collect()]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let gx2 = srs.X2P[0] * inputs[1].raw[0] + srs.Y2P * inputs[1].r;
    let C = inputs[0].g1 * inputs[1].r + inputs[1].g1 * inputs[0].r + srs.Y1P * (inputs[0].r * inputs[1].r) - srs.X1P[0] * outputs[0].r;
    return (vec![C], vec![gx2]);
  }
  fn verify(
    &self,
    srs: &SRS,
    _model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    // Verify f(x)*g(x)=h(x)
    let lhs = Bn254::pairing(inputs[0].g1, proof.1[0]);
    let rhs = Bn254::pairing(outputs[0].g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
    // Verify gx2
    let lhs = Bn254::pairing(inputs[1].g1, srs.X2A[0]);
    let rhs = Bn254::pairing(srs.X1A[0], proof.1[0]);
    assert!(lhs == rhs);
  }
}
pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1, 1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().zip(inputs[1]).map(|(x, y)| *x * *y).collect()]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let N = inputs[0].raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = util::msm::<G2Projective>(&srs.X2A, &inputs[1].poly.coeffs) + srs.Y2P * inputs[1].r;
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&outputs[0].poly).divide_by_vanishing_poly(domain).unwrap().0;

    // Blinding
    let mut rng = StdRng::from_entropy();
    let r = Fr::rand(&mut rng);
    let tx = util::msm::<G1Projective>(&srs.X1A, &t.coeffs) + srs.Y1P * r;
    let C = (inputs[0].g1 * inputs[1].r) + (inputs[1].g1 * inputs[0].r) + (srs.Y1P * (inputs[0].r * inputs[1].r))
      - (srs.X1P[0] * outputs[0].r)
      - ((srs.X1P[N] - srs.X1P[0]) * r);
    return (vec![tx, C], vec![gx2]);
  }
  fn verify(
    &self,
    srs: &SRS,
    _model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bn254::pairing(inputs[0].g1, proof.1[0]) - Bn254::pairing(outputs[0].g1, srs.X2A[0]);
    let rhs = Bn254::pairing(proof.0[0], srs.X2A[inputs[0].len] - srs.X2A[0]) + Bn254::pairing(proof.0[1], srs.Y2A);
    assert!(lhs == rhs);
    // Verify gx2
    let lhs = Bn254::pairing(inputs[1].g1, srs.X2A[0]);
    let rhs = Bn254::pairing(srs.X1A[0], proof.1[0]);
    assert!(lhs == rhs);
  }
}
