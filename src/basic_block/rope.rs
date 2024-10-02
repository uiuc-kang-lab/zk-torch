use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};

#[derive(Debug)]
pub struct RoPEBasicBlock {
  pub token_i: usize,
  pub output_SF: usize,
}

impl BasicBlock for RoPEBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let mut r1 = vec![];
    let mut r2 = vec![];
    for i in 0..64 {
      let x = (self.token_i as f64) / (10000_f64.powf((i as f64) / 64_f64));
      let mut a = x.cos();
      let mut b = x.sin();
      a *= (1 << self.output_SF) as f64;
      b *= (1 << self.output_SF) as f64;
      let a = Fr::from(a.round() as i128);
      let b = Fr::from(b.round() as i128);
      r1.push(a);
      r2.push(b);
    }
    Ok(vec![arr1(&r1).into_dyn(), arr1(&r2).into_dyn()])
  }
}
