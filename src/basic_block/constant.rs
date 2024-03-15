use super::BasicBlock;
use ark_bn254::Fr;

pub struct ConstBasicBlock;
impl BasicBlock for ConstBasicBlock {
  fn run(&self, model: &Vec<Fr>, _inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![model.clone()]
  }
}
