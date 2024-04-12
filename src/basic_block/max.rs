use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{arr0, ArrayD};

pub struct MaxBasicBlock;
impl BasicBlock for MaxBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    vec![arr0(inputs[0].fold(Fr::zero(), |max, x| {
      if *x < Fr::from(1 << 28) && *x > max {
        return *x;
      } else {
        return max;
      }
    }))
    .into_dyn()]
  }
}
