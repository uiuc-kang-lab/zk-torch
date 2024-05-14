use super::{BasicBlock, Data, DataEnc, PairingCheck, SRS};
use ark_bn254::{G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::ArrayD;
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct EqBasicBlock;
impl BasicBlock for EqBasicBlock {
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[1].ndim() == 1);
    // Blinding
    let C = srs.X1P[0] * (inputs[0][0].r - inputs[1][0].r);
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
  ) -> Vec<PairingCheck> {
    // Verify f(x)+g(x)=h(x)
    vec![vec![
      (inputs[0][0].g1, srs.X2A[0]),
      (-inputs[1][0].g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]]
  }
}
