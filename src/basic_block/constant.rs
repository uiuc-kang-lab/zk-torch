use super::{BasicBlock, BasicBlockType};
use ark_bn254::Fr;
use ndarray::ArrayD;

#[derive(Debug)]
pub struct ConstBasicBlock {
  pub name: String,
}

impl BasicBlock for ConstBasicBlock {
  fn block_type(&self) -> Result<BasicBlockType, String> {
    Ok(BasicBlockType::Constant)
  }

  fn weights_name(&self) -> Result<String, String> {
    Ok(self.name.clone())
  }

  fn run(&self, weights: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![weights.clone()]
  }
}
