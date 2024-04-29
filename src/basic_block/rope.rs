use super::{BasicBlock, BasicBlockType};
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};

pub struct RoPEBasicBlock {
  pub token_i: usize,
  pub output_SF: usize,
}

impl BasicBlock for RoPEBasicBlock {
  fn block_type(&self) -> BasicBlockType {
    BasicBlockType::RoPE
  }

  fn name(&self) -> String {
    format!("RoPE[token_i: {}, output_SF: {}]", self.token_i, self.output_SF)
  }

  fn run(&self, _weights: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
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
