use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;
use rayon::prelude::*;

#[derive(Debug)]
pub struct ClipBasicBlock {
  pub min: f32,
  pub max: f32,
}
impl BasicBlock for ClipBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();
    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let mut x = util::fr_to_int(*x) as f32;
        x = x.max(self.min).min(self.max);
        Fr::from(x.round() as i32)
      })
      .collect::<Vec<_>>();
    vec![ArrayD::from_shape_vec(shape, out).unwrap()]
  }
}
