use super::BasicBlock;
use ark_bn254::Fr;

pub struct RoPEBasicBlock {
  pub token_i: usize,
  pub output_SF: usize,
}
impl BasicBlock for RoPEBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, _inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
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
    vec![r1, r2]
  }
}
