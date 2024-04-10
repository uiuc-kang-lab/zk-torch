use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;

pub struct DivConstBasicBlock {
  pub c: usize,
}
impl BasicBlock for DivConstBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let mut r = vec![];
    for x in inputs[0].iter() {
      let mut x = util::fr_to_int(*x) as f32;
      x /= self.c as f32;
      let x = Fr::from(x.round() as i32);
      r.push(x);
    }
    vec![r]
  }
}

pub struct DivScalarBasicBlock {
  pub output_SF: usize,
}
impl BasicBlock for DivScalarBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![1])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    let SF = self.output_SF as i32;
    let mut div = vec![];
    let mut rem = vec![];
    let y = util::fr_to_int(inputs[1][0]); //Assumes this is positive
    for x in inputs[0].iter() {
      let x = util::fr_to_int(*x);
      let mut z = (2 * x * SF + y) / (2 * y);
      let mut r = (2 * x * SF + y) % (2 * y);
      if r < 0 {
        z -= 1;
        r += 2 * y;
      }
      div.push(Fr::from(z));
      rem.push(Fr::from(r));
    }
    vec![div, rem]
  }
}
