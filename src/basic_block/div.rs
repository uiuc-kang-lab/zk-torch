use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use rayon::prelude::*;

#[derive(Debug)]
pub struct DivScalarBasicBlock {
  pub output_SF: usize,
}

impl BasicBlock for DivScalarBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[1].len() == 1);
    let SF = self.output_SF as i64;
    let y = util::fr_to_int(inputs[1][0]) as i64;
    assert!(y > 0);
    let (div, rem): (Vec<_>, Vec<_>) = util::array_into_iter(inputs[0])
      .map(|x| {
        let x = util::fr_to_int(*x) as i64;
        let mut z = (2 * x * SF + y) / (2 * y);
        let mut r = (2 * x * SF + y) % (2 * y);
        if r < 0 {
          z -= 1;
          r += 2 * y;
        }
        (Fr::from(z), Fr::from(r))
      })
      .unzip();
    vec![arr1(&div).into_dyn(), arr1(&rem).into_dyn()]
  }
}

#[derive(Debug)]
pub struct DivConstBasicBlock {
  pub c: f32,
}

impl BasicBlock for DivConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();

    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let mut x = util::fr_to_int(*x) as f32;
        x /= self.c;
        Fr::from(x.round() as i64)
      })
      .collect::<Vec<_>>();

    vec![ArrayD::from_shape_vec(shape, out).unwrap()]
  }
}

#[derive(Debug)]
pub struct ModConstBasicBlock {
  pub c: u32,
}
impl BasicBlock for ModConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();

    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let x = util::fr_to_int(*x) as u32;
        Fr::from((x % self.c) as i64)
      })
      .collect::<Vec<_>>();

    vec![ArrayD::from_shape_vec(shape, out).unwrap()]
  }
}
