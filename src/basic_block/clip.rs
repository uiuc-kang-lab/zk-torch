use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr1, ArrayD, IxDyn};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct ClipBasicBlock {
  pub min: Fr,
  pub max: Fr,
}

// ClipBasicBlock is a basic block that clips the input tensor to a specified range.
// This block requires formal proving, which will be implemented in the next snippet after MaxBasicBlock.
impl BasicBlock for ClipBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    vec![inputs[0].mapv(|x| x.max(self.min).min(self.max))]
  }
}
