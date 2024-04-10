use super::{BasicBlock, Data, DataEnc, SRS};
use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use rand::rngs::StdRng;

pub struct EqBasicBlock;
impl BasicBlock for EqBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1, 1])
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    // Blinding
    let C = srs.X1P[0] * (inputs[0].r - inputs[1].r);
    (vec![C], Vec::new())
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
    // Verify f(x)+g(x)=h(x)
    let lhs = Bn254::pairing(inputs[0].g1, srs.X2A[0]);
    let rhs = Bn254::pairing(inputs[1].g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
  }
}
