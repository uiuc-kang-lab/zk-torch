use super::{BasicBlock, Data, DataEnc};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use rand::rngs::StdRng;

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(&self, _model: &Vec<Fr>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().zip(inputs[1].iter()).map(|(x,y)|*x+*y).collect()]
  }
  fn prove(
    &mut self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    // Blinding
    let C = srs.0[0] * (inputs[0].r + inputs[1].r - outputs[0].r);
    (vec![C.into()], Vec::new())
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    // Verify f(x)+g(x)=h(x)
    let lhs = Bn254::pairing(inputs[0].g1 + inputs[1].g1, srs.1[0]);
    let rhs = Bn254::pairing(outputs[0].g1, srs.1[0]) + Bn254::pairing(proof.0[0], srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
  }
}
