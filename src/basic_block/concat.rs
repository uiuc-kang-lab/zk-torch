use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{Array1, ArrayD, Axis};
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

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let mut r = inputs[0].clone().to_owned();
    if inputs[0].ndim() == 0 {
      let r_vec = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<Data>>();
      r = Array1::from_vec(r_vec).into_dyn();
    } else {
      for input in inputs.iter().skip(1) {
        r = ndarray::concatenate(Axis(self.axis), &[r.view(), input.view()]).unwrap();
      }
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
    for i in 0..inputs.len() {
      inputs[i].iter().zip(outputs[0].index_axis(Axis(self.axis), i).iter()).for_each(|(input, output)| {
        assert!(input == output);
      });
    }
    vec![]
  }
}
