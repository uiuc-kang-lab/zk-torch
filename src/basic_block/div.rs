use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};

#[derive(Debug)]
pub struct DivScalarBasicBlock {
  pub output_SF: usize,
}

impl BasicBlock for DivScalarBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[1].len() == 1);
    let SF = self.output_SF as i32;
    let mut div = vec![];
    let mut rem = vec![];
    let y = util::fr_to_int(inputs[1][0]);
    assert!(y > 0);
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
    vec![arr1(&div).into_dyn(), arr1(&rem).into_dyn()]
  }
}
