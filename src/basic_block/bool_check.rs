use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

// BooleanCheckBasicBlock is a basic block that checks if all elements are 0 or 1
// This block is used to ensure that the output of a model is a boolean tensor during model compilation
#[derive(Debug)]
pub struct BooleanCheckBasicBlock;
impl BasicBlock for BooleanCheckBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // check if all elements are 0 or 1
    assert!(inputs.iter().all(|x| x.iter().all(|y| {
      let y_int = util::fr_to_int(*y);
      y_int == 0 || y_int == 1
    })));
    vec![]
  }
}
