use super::BasicBlock;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};

#[derive(Debug)]
pub struct RoPEBasicBlock {
  pub token_i: usize,
  pub output_SF: usize,
}

impl BasicBlock for RoPEBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let mut r1 = vec![];
    let mut r2 = vec![];
    for i in 0..64 {
      let x = (self.token_i as f32) / (10000_f32.powf((i as f32) / 64_f32));
      let mut a = x.cos();
      let mut b = x.sin();
      a *= (1 << self.output_SF) as f32;
      b *= (1 << self.output_SF) as f32;
      let a = Fr::from(a.round() as i32);
      let b = Fr::from(b.round() as i32);
      r1.push(a);
      r2.push(b);
    }
    vec![arr1(&r1).into_dyn(), arr1(&r2).into_dyn()]
  }
}
