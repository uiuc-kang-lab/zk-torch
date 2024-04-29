use super::{BasicBlock, BasicBlockType};
use ark_bn254::Fr;
use ndarray::ArrayD;

pub struct ConstBasicBlock {
  pub name: String,
}

impl BasicBlock for ConstBasicBlock {
  fn name(&self) -> String {
    format!("Constant-{}", self.name)
  }

  fn weights_name(&self) -> String {
    self.name.clone()
  }

  fn run(&self, weights: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![weights.clone()]
  }
}
