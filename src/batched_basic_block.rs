use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr1, ArrayD, Axis};
use rand::rngs::StdRng;

fn arr_to_vec<P: Clone>(a: &Vec<&ArrayD<P>>) -> Vec<Vec<P>> {
  a.iter()
    .map(|x| {
      if x.ndim() == 1 {
        vec![(*x).clone().into_owned().into_raw_vec()]
      } else {
        x.axis_iter(Axis(x.ndim() - 1)).map(|y| y.into_owned().into_raw_vec()).collect::<Vec<_>>()
      }
    })
    .flatten()
    .collect()
}
fn arr_flatten<P: Clone>(a: &Vec<&ArrayD<P>>) -> Vec<P> {
  a.iter().map(|x| (*x).clone().into_raw_vec()).flatten().collect()
}
fn vec_to_arr<P: Clone>(a: Vec<Vec<P>>) -> Vec<ArrayD<P>> {
  a.iter().map(|x| arr1(x).into_dyn()).collect()
}
pub struct BatchedBasicBlock {
  pub basic_block: Box<dyn BasicBlock>,
}
impl BatchedBasicBlock {
  pub fn run(&self, model: &Vec<&ArrayD<Fr>>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert_eq!(
      (model.iter().map(|x| x.ndim()).collect(), inputs.iter().map(|x| x.ndim()).collect()),
      self.basic_block.get_dims()
    );
    let model = arr_to_vec(model);
    let model = model.iter().map(|x| x).collect();
    let inputs = arr_to_vec(inputs);
    let inputs = inputs.iter().map(|x| x).collect();
    let outputs = self.basic_block.run(&model, &inputs);
    vec_to_arr(outputs)
  }
  pub fn setup(&self, srs: &SRS, model: &Vec<&ArrayD<Data>>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert_eq!(model.iter().map(|x| x.ndim() + 1).collect::<Vec<_>>(), self.basic_block.get_dims().0);
    let model = arr_flatten(model);
    let model = model.iter().map(|x| x).collect();
    self.basic_block.setup(srs, &model)
  }
  pub fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert_eq!(
      (
        model.iter().map(|x| x.ndim() + 1).collect(),
        inputs.iter().map(|x| x.ndim() + 1).collect()
      ),
      self.basic_block.get_dims()
    );
    let model = arr_flatten(model);
    let model = model.iter().map(|x| x).collect();
    let inputs = arr_flatten(inputs);
    let inputs = inputs.iter().map(|x| x).collect();
    let outputs = arr_flatten(outputs);
    let outputs = outputs.iter().map(|x| x).collect();
    self.basic_block.prove(srs, setup, &model, &inputs, &outputs, rng)
  }
  pub fn verify(
    &self,
    srs: &SRS,
    model: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    assert_eq!(
      (
        model.iter().map(|x| x.ndim() + 1).collect(),
        inputs.iter().map(|x| x.ndim() + 1).collect()
      ),
      self.basic_block.get_dims()
    );
    let model = arr_flatten(model);
    let model = model.iter().map(|x| x).collect();
    let inputs = arr_flatten(inputs);
    let inputs = inputs.iter().map(|x| x).collect();
    let outputs = arr_flatten(outputs);
    let outputs = outputs.iter().map(|x| x).collect();
    self.basic_block.verify(srs, &model, &inputs, &outputs, proof, rng)
  }
}
