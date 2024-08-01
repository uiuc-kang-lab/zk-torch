use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

// perform element-wise less than comparison for two 1-d arrays
#[derive(Debug)]
pub struct LessBasicBlock;
impl BasicBlock for LessBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let mut r = ArrayD::zeros(IxDyn(&[std::cmp::max(inputs[0].len(), inputs[1].len())]));
    if inputs[0].len() == 1 && inputs[1].ndim() > 0 {
      azip!((r in &mut r, &x in inputs[1]) *r = Fr::from((util::fr_to_int(x) >= util::fr_to_int(*inputs[0].first().unwrap())) as i32));
    } else if inputs[1].len() == 1 {
      azip!((r in &mut r, &x in inputs[0]) *r = Fr::from((util::fr_to_int(x) < util::fr_to_int(*inputs[1].first().unwrap())) as i32));
    } else {
      azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = Fr::from((util::fr_to_int(x) < util::fr_to_int(y)) as i32));
    }
    vec![r]
  }
}
