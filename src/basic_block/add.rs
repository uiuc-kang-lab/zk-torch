use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Zero;
use ark_poly::univariate::DensePolynomial;
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

// This basic block is used to add two inputs together
// Note: The inputs are expected to have at most 1 dimension
#[derive(Debug)]
pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let mut r = ArrayD::zeros(IxDyn(&[std::cmp::max(inputs[0].len(), inputs[1].len())]));
    if inputs[0].len() == 1 && inputs[1].ndim() > 0 {
      azip!((r in &mut r, &x in inputs[1]) *r = x + inputs[0].first().unwrap());
    } else if inputs[1].len() == 1 {
      azip!((r in &mut r, &x in inputs[0]) *r = x + inputs[1].first().unwrap());
    } else {
      azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x + y);
    }
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let a = &inputs[0].first().unwrap();
    let b = &inputs[1].first().unwrap();
    vec![arr0(Data {
      raw: outputs[0].clone().into_raw_vec(),
      poly: (&a.poly) + (&b.poly),
      g1: a.g1 + b.g1,
      r: a.r + b.r,
    })
    .into_dyn()]
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
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    let c = outputs[0].first().unwrap();
    // Verify f(x)+g(x)=h(x)
    assert!(a.g1 + b.g1 == c.g1);
    vec![]
  }
}

// This basic block is used to add multiple inputs together
// Note: The inputs are expected to have the same shape
#[derive(Debug)]
pub struct BatchAddBasicBlock;
impl BasicBlock for BatchAddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    // The inputs are expected to have the same shape
    assert!(inputs.iter().all(|x| x.shape() == inputs[0].shape()));
    let mut r = ArrayD::zeros(IxDyn(&[inputs[0].len()]));
    for i in 0..inputs.len() {
      azip!((r in &mut r, &x in inputs[i]) *r = *r + x);
    }
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    vec![arr0(Data {
      raw: outputs[0].clone().into_raw_vec(),
      poly: inputs.iter().fold(DensePolynomial::zero(), |acc, x| acc + x.first().unwrap().poly.clone()),
      g1: inputs.iter().fold(G1Projective::zero(), |acc, x| acc + x.first().unwrap().g1),
      r: inputs.iter().fold(Fr::zero(), |acc, x| acc + x.first().unwrap().r),
    })
    .into_dyn()]
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
    let inputs_g1 = inputs.iter().fold(G1Projective::zero(), |acc, x| acc + x.first().unwrap().g1);
    let c_g1 = outputs[0].first().unwrap().g1;
    // Verify f1(x)+f2(x)+...+fn(x)=h(x)
    assert!(inputs_g1 == c_g1);
    vec![]
  }
}
