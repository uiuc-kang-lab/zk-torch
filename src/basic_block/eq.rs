use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ndarray::ArrayD;
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct EqBasicBlock;
impl BasicBlock for EqBasicBlock {
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    // Blinding
    let C = srs.X1P[0] * (inputs[0].first().unwrap().r - inputs[1].first().unwrap().r);
    (vec![C], Vec::new(), Vec::new())
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    // Verify f(x)+g(x)=h(x)
    vec![vec![
      (inputs[0].first().unwrap().g1, srs.X2A[0]),
      (-inputs[1].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]]
  }
}
