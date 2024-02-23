use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{azip, ArrayD};

pub struct ReLUBasicBlock;
impl BasicBlock for ReLUBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> ArrayD<Fr> {
    let mut r = ArrayD::zeros(inputs[0].shape());
    azip!((&x in inputs[0], z in &mut r) if x < Fr::from(1<<28){*z = x}else{*z = Fr::zero()});
    r
  }
}
