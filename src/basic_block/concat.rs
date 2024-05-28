use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{ArrayD, Axis};
use rand::rngs::StdRng;

// only support concat over dim 0 for now
#[derive(Debug)]
pub struct ConcatBasicBlock {
  pub axis: usize,
}

impl BasicBlock for ConcatBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    for input in inputs.iter() {
      assert!(inputs[0].shape()[1..] == input.shape()[1..]);
    }
    let mut r = inputs[0].clone().to_owned();
    for input in inputs.iter().skip(1) {
      r = ndarray::concatenate(Axis(self.axis), &[r.view(), input.view()]).unwrap();
    }
    vec![r]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    for input in inputs.iter() {
      assert!(inputs[0].shape()[1..] == input.shape()[1..]);
    }
    assert!(outputs[0].shape()[0] == inputs.len());
    let mut r = inputs[0].clone().to_owned();
    for input in inputs.iter().skip(1) {
      r = ndarray::concatenate(Axis(self.axis), &[r.view(), input.view()]).unwrap();
    }
    vec![r]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    for input in inputs.iter() {
      assert!(inputs[0].shape()[1..] == input.shape()[1..]);
    }
    assert!(outputs[0].shape()[0] == inputs.len());
    for i in 0..inputs.len() {
      inputs[i].iter().zip(outputs[0].index_axis(Axis(self.axis), i).iter()).for_each(|(input, output)| {
        assert!(input.g1 == output.g1);
      });      
    }
    vec![]
  }
}