use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ndarray::{ArrayD, IxDyn, Zip};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct EqBasicBlock;
impl BasicBlock for EqBasicBlock {
  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
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
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    // Verify f(x)+g(x)=h(x)
    vec![vec![
      (inputs[0].first().unwrap().g1, srs.X2A[0]),
      (-inputs[1].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]]
  }
}

// ElementwiseEqBasicBlock is a basic block that performs elementwise equality comparison.
#[derive(Debug)]
pub struct ElementwiseEqBasicBlock;
impl BasicBlock for ElementwiseEqBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let mut r = ArrayD::zeros(IxDyn(&[std::cmp::max(inputs[0].len(), inputs[1].len())]));
    // broadcast inputs[0] to compare with each element in inputs[1]
    if inputs[0].len() == 1 && inputs[1].ndim() > 0 {
      Zip::from(r.view_mut())
        .and(inputs[1].view())
        .for_each(|r, &x| *r = (util::fr_to_int(x) == util::fr_to_int(*inputs[0].first().unwrap())) as u8);
    // broadcast inputs[1] to compare with each element in inputs[0]
    } else if inputs[1].len() == 1 && inputs[0].ndim() > 0 {
      Zip::from(r.view_mut())
        .and(inputs[0].view())
        .for_each(|r, &x| *r = (util::fr_to_int(x) == util::fr_to_int(*inputs[1].first().unwrap())) as u8);
    // elementwise comparison
    } else {
      Zip::from(r.view_mut())
        .and(inputs[0].view())
        .and(inputs[1].view())
        .for_each(|r, &x, &y| *r = (util::fr_to_int(x) == util::fr_to_int(y)) as u8);
    }

    vec![r.map(|&x| Fr::from(x)).into_dyn()]
  }
}
