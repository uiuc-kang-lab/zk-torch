use super::BasicBlock;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;

pub struct ExpBasicBlock {
  pub input_SF: usize,
  pub output_SF: usize,
}
impl BasicBlock for ExpBasicBlock {
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut r = vec![];
    for x in inputs[0].iter() {
      let x = *x;
      let mut x = if x < Fr::from(1 << 28) {
        x.into_bigint().0[0] as f32
      } else {
        -((-x).into_bigint().0[0] as f32)
      };
      x /= (1 << self.input_SF) as f32;
      x = x.exp();
      x *= (1 << self.output_SF) as f32;
      let x = Fr::from(x as i32);
      r.push(x);
    }
    vec![r]
  }
}
