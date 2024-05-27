use super::{BasicBlock, Data, SRS};
use ark_bn254::Fr;
use ndarray::ArrayD;

#[derive(Debug)]
pub struct TransposeBasicBlock {
  pub perm: Vec<usize>,
}

impl BasicBlock for TransposeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    assert!(*self.perm.last().unwrap() == self.perm.len() - 1);
    vec![inputs[0].view().permuted_axes(&self.perm[..]).to_owned()]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let n = self.perm.len();
    vec![inputs[0].view().permuted_axes(&self.perm[..n - 1]).to_owned()]
  }
}
