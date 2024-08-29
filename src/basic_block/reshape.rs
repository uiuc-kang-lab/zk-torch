use crate::util;

use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::ArrayD;
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct ReshapeBasicBlock {
  pub shape: Vec<usize>,
}

impl BasicBlock for ReshapeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    assert!(inputs[0].shape().last() == self.shape.last());
    vec![inputs[0].view().into_shape(&self.shape[..]).unwrap().to_owned()]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let n = self.shape.len();
    vec![inputs[0].view().into_shape(&self.shape[..n - 1]).unwrap().to_owned()]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let n = self.shape.len();
    let view = inputs[0].view().into_shape(&self.shape[..n - 1]).unwrap();
    assert!(outputs[0] == view);

    vec![]
  }
}
