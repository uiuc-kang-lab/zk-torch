use super::BasicBlock;
use ark_bn254::Fr;
use ndarray::ArrayD;

#[derive(Debug)]
pub struct ConstBasicBlock;
impl BasicBlock for ConstBasicBlock {
  fn run(&self, model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![model.clone()]
  }
}
