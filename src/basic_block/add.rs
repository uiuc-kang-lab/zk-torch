use super::{BasicBlock, Data, DataEnc};
use ark_bn254::{Fr, G1Affine, G2Affine};
use rand::Rng;

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(_model: &Data, inputs: &Vec<Vec<Fr>>) -> Vec<Fr> {
    let mut r = Vec::new();
    for i in 0..inputs[0].len() {
      r.push(inputs[0][i] + inputs[1][i]);
    }
    return r;
  }
  fn verify<R: Rng>(
    _srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut R,
  ) {
    // Verify f(x)+g(x)=h(x)
    let lhs = inputs[0].g1 + inputs[1].g1;
    let rhs = output.g1;
    assert!(lhs == rhs);
  }
}
