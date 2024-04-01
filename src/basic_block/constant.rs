use super::BasicBlock;
use ark_bn254::Fr;

pub struct ConstBasicBlock;
impl BasicBlock for ConstBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![2], vec![])
  }
  fn run(&self, model: &Vec<&Vec<Fr>>, _inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    model.iter().map(|x| (*x).clone()).collect()
  }
}
