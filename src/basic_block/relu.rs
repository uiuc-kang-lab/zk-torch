use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;

pub struct ReLUBasicBlock;
impl BasicBlock for ReLUBasicBlock {
  fn name(&self) -> &str {
    "ReLU"
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().map(|x| if *x < Fr::from(1 << 28) { *x } else { Fr::zero() }).collect()]
  }
}
