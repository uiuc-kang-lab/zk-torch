use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{Array1, ArrayD, Axis};
use rand::rngs::StdRng;

// support concat over any dim except for the last
#[derive(Debug)]
pub struct ConcatBasicBlock {
  pub axis: usize,
}

impl BasicBlock for ConcatBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(self.axis != inputs[0].shape().len() - 1);
    let r = ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
    vec![r]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    if inputs[0].ndim() == 0 {
      let r_vec = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<Data>>();
      let r = Array1::from_vec(r_vec).into_dyn();
      vec![r]
    } else {
      let r = ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
      vec![r]
    }
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
    if inputs[0].ndim() == 0 {
      let r = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<DataEnc>>();
      let r_enc = outputs[0];
      for i in 0..r.len() {
        assert!(r[i] == r_enc[i]);
      }
    } else {
      assert!(ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()) == Ok(outputs[0].clone()));
    }
    vec![]
  }
}
