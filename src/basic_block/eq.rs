use crate::graph::SetupType;

use super::{BasicBlock, BasicBlockType, Data, DataEnc, SRS};
use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub struct EqBasicBlock;

impl BasicBlock for EqBasicBlock {
  fn block_type(&self) -> BasicBlockType {
    BasicBlockType::Eq
  }

  fn name(&self) -> String {
    "Eq".to_string()
  }

  fn prove(
    &mut self,
    srs: &SRS,
    _setup: &SetupType,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    // Blinding
    let C = srs.X1P[0] * (a.r - b.r);
    (vec![C], Vec::new())
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    // Verify f(x)+g(x)=h(x)
    let lhs = Bn254::pairing(a.g1, srs.X2A[0]);
    let rhs = Bn254::pairing(b.g1, srs.X2A[0]) + Bn254::pairing(proof.0[0], srs.Y2A);
    assert!(lhs == rhs);
  }
}
