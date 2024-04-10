use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;

pub struct ReLUBasicBlock;
impl BasicBlock for ReLUBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().map(|x| if *x < Fr::from(1 << 28) { *x } else { Fr::zero() }).collect()]
  }
}
