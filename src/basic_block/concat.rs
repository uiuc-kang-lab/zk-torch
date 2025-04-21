use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_std::Zero;
use ndarray::{indices, Array1, ArrayD, Axis, Slice, SliceInfoElem};
use rand::rngs::StdRng;
use std::fmt::Debug;

fn concatenate_sliced<G: Clone + Debug>(inputs: &Vec<&ArrayD<G>>, axis: usize, input_shapes: &Vec<Vec<usize>>) -> ArrayD<G> {
  let sliced_views: Vec<_> = inputs
    .iter()
    .zip(input_shapes.iter())
    .map(|(x, shape)| {
      let slice_info: Vec<SliceInfoElem> = (0..x.ndim())
        .map(|i| {
          if i < shape.len() - 1 {
            SliceInfoElem::from(..shape[i])
          } else {
            SliceInfoElem::from(..)
          }
        })
        .collect();
      x.slice(slice_info.as_slice())
    })
    .collect();

  ndarray::concatenate(Axis(axis), &sliced_views).unwrap()
}

// support concat over any dim except for the last
#[derive(Debug)]
pub struct ConcatBasicBlock {
  pub axis: usize,
  pub input_shapes: Vec<Vec<usize>>,
}

impl BasicBlock for ConcatBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(self.axis != inputs[0].shape().len() - 1);
    let r = concatenate_sliced(inputs, self.axis, &self.input_shapes);
    Ok(vec![util::pad_to_pow_of_two(&r, &Fr::zero())])
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    if inputs[0].ndim() == 0 {
      let r_vec = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<Data>>();
      let r = Array1::from_vec(r_vec).into_dyn();
      vec![r]
    } else {
      let N = inputs[0].first().unwrap().raw.len();
      let r = concatenate_sliced(inputs, self.axis, &self.input_shapes);
      let data = Data::new(srs, &vec![Fr::zero(); N]);
      vec![util::pad_to_pow_of_two(&r, &data)]
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
        assert!(r[i] == r_enc[i], "Mismatch at index {:?}", i);
      }
    } else {
      let r = concatenate_sliced(inputs, self.axis, &self.input_shapes);
      let r_enc = outputs[0];

      for indices in ndarray::indices(r.shape()) {
        let r_val = &r[&indices];
        let r_enc_val = &r_enc[&indices];
        assert!(r_val == r_enc_val, "Mismatch at indices {:?}", indices);
      }
    }
    vec![]
  }
}
