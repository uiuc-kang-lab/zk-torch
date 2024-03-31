use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;

pub struct PowBasicBlock {
  pub input_SF: usize,
  pub output_SF: usize,
}
impl BasicBlock for PowBasicBlock {
  // FIXME: currently only supports squaring
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut r = vec![];
    for x in inputs[0].iter() {
      let mut x = util::fr_to_int(*x) as f32;
      x /= self.input_SF as f32;
      x = x.powf(2.);
      x *= self.output_SF as f32;
      let x = Fr::from(x.round() as i32);
      r.push(x);
    }
    vec![r]
  }
}
