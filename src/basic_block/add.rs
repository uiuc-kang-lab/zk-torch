use super::{BasicBlock, Data, DataEnc, SRS};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ndarray::{azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let mut r = ArrayD::zeros(IxDyn(&[std::cmp::max(inputs[0].len(), inputs[1].len())]));
    if inputs[0].len() == 1 {
      azip!((r in &mut r, &x in inputs[1]) *r = x + inputs[0].first().unwrap());
    } else if inputs[1].len() == 1 {
      azip!((r in &mut r, &x in inputs[0]) *r = x + inputs[1].first().unwrap());
    } else {
      azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x + y);
    }
    vec![r]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    let c = outputs[0].first().unwrap();
    // Blinding
    let C = srs.X1P[0] * (a.r + b.r - c.r);
    (vec![C], Vec::new())
  }
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    let c = outputs[0].first().unwrap();
    // Verify f(x)+g(x)=h(x)
    let lhs = Bn254::pairing(a.g1 + b.g1, srs.X2A[0]);
    let rhs = Bn254::pairing(c.g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
  }
}
