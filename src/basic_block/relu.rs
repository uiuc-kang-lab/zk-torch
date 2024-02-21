use super::{BasicBlock, Data, DataEnc};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::BigInt;
use ark_std::Zero;
use ndarray::{azip, ArrayD};
use rand::rngs::StdRng;

pub struct ReLUBasicBlock;
impl BasicBlock for ReLUBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<ArrayD<Fr>>) -> ArrayD<Fr> {
    let mut r = ArrayD::zeros(inputs[0].shape());
    azip!((&x in &inputs[0], z in &mut r) if x.0 < BigInt::new([1<<28,0,0,0]){*z = x}else{*z = Fr::zero()});
    r
  }
}
