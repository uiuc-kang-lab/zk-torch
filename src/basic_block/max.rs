use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;

pub struct MaxBasicBlock;
impl BasicBlock for MaxBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![2])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut max = Fr::zero();
    for i in 0..inputs.len() {
      for j in 0..inputs[0].len() {
        if inputs[i][j] < Fr::from(1 << 28) && inputs[i][j] > max {
          max = inputs[i][j];
        }
      }
    }
    vec![vec![max]]
  }
}
