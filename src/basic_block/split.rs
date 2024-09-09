use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{ArrayD, Axis};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct SplitBasicBlock {
  pub axis: usize,
  pub split: Vec<usize>,
}

impl BasicBlock for SplitBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    assert!(self.axis < inputs[0].ndim() - 1);
    assert!(inputs[0].shape()[self.axis] == self.split.iter().sum::<usize>());
    let mut r = vec![];
    // use split_at
    let mut b = inputs[0].view();
    for &s in self.split.iter() {
      let (a, remaining) = b.split_at(Axis(self.axis), s);
      b = remaining;
      r.push(a.to_owned());
    }
    r
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let mut r = vec![];
    // use split_at
    let mut b = inputs[0].view();
    for &s in self.split.iter() {
      let (a, remaining) = b.split_at(Axis(self.axis), s);
      b = remaining;
      r.push(a.to_owned());
    }
    r
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
    let mut b = inputs[0].view();
    for i in 0..outputs.len() {
      let (a, remaining) = b.split_at(Axis(self.axis), self.split[i]);
      b = remaining;
      outputs[i].iter().zip(a.iter()).for_each(|(input, output)| {
        assert!(input == output);
      });
    }
    vec![]
  }
}
