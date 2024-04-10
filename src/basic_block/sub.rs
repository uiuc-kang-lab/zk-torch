use super::{BasicBlock, Data, DataEnc, SRS};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use rand::rngs::StdRng;

pub struct SubBasicBlock;
impl BasicBlock for SubBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1, 1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut r = vec![];
    let m = ark_std::cmp::max(inputs[0].len(), inputs[1].len());
    for i in 0..m {
      if inputs[0].len() <= i {
        r.push(inputs[0][0] - inputs[1][i]);
      } else if inputs[1].len() <= i {
        r.push(inputs[0][i] - inputs[1][0]);
      } else {
        r.push(inputs[0][i] - inputs[1][i]);
      }
    }
    vec![r]
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
    let C = srs.X1P[0] * (inputs[0].r - inputs[1].r - outputs[0].r);
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
    // Verify f(x)-g(x)=h(x)
    let lhs = Bn254::pairing(inputs[0].g1 - inputs[1].g1, srs.X2A[0]);
    let rhs = Bn254::pairing(outputs[0].g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
  }
}
