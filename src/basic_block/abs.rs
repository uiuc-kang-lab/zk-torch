use super::BasicBlock;
use ark_bn254::Fr;
use ark_ff::Field;
use ark_std::Zero;

pub struct AbsBasicBlock;
impl BasicBlock for AbsBasicBlock {
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![inputs[0].iter().map(|x| if *x < Fr::from(1 << 28) { *x } else { *x.clone().neg_in_place() }).collect()]
  }
}
