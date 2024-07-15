use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr1, ArrayD, IxDyn};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct RangeBasicBlock {
  pub start: Fr,
  pub limit: Fr,
  pub delta: Fr,
}
impl BasicBlock for RangeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let mut r = vec![];
    let mut x = self.start;
    while x < self.limit {
      r.push(x);
      x += self.delta;
    }
    vec![arr1(&r).into_dyn()]
  }
}
