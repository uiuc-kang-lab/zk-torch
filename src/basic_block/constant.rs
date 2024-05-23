use super::{BasicBlock, Data, SRS};
use ark_bn254::Fr;
use ndarray::ArrayD;

#[derive(Debug)]
pub struct ConstBasicBlock;
impl BasicBlock for ConstBasicBlock {
  fn run(&self, model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![model.clone()]
  }
  fn encodeOutputs(&self, _srs: &SRS, model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    vec![model.clone()]
  }
}
