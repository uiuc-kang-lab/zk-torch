use super::BasicBlock;
use ark_bn254::Fr;
use ark_std::Zero;

pub struct LogBasicBlock {
  input_SF: usize,
  output_SF: usize,
}
impl BasicBlock for LogBasicBlock {
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut r = vec![];
    for x in inputs[0].iter() {
      let x = *x;
      let mut x = if x < Fr::from(1 << 28) {
        x.0 .0[0] as f32
      } else {
        -((-x).0 .0[0] as f32)
      };
      x /= self.input_SF as f32;
      x = x.ln();
      x *= self.output_SF as f32;
      let x = Fr::from(x.round() as i32);
      r.push(x);
    }
    vec![r]
  }
}
