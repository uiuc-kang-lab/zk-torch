use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use ndarray::{azip, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct MulConstBasicBlock {
  pub c: usize,
}

impl BasicBlock for MulConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    vec![inputs[0].map(|x| *x * Fr::from(self.c as u32))]
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let C = srs.X1P[0] * (Fr::from(self.c as u32) * inputs[0].first().unwrap().r - outputs[0].first().unwrap().r);
    return (vec![C], vec![], Vec::new());
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    vec![vec![
      (inputs[0].first().unwrap().g1, (srs.X2P[0] * Fr::from(self.c as u32)).into()),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]]
  }
}

#[derive(Debug)]
pub struct MulScalarBasicBlock;
impl BasicBlock for MulScalarBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].len() == 1);
    vec![inputs[0].map(|x| *x * inputs[1].first().unwrap())]
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0 = &inputs[0].first().unwrap();
    let inp1 = &inputs[1].first().unwrap();
    let out = &outputs[0].first().unwrap();
    let gx2 = srs.X2P[0] * inp1.raw[0] + srs.Y2P * inp1.r;
    let C = inp0.g1 * inp1.r + inp1.g1 * inp0.r + srs.Y1P * (inp0.r * inp1.r) - srs.X1P[0] * out.r;
    return (vec![C], vec![gx2], Vec::new());
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    // Verify f(x)*g(x)=h(x)
    checks.push(vec![
      (inputs[0].first().unwrap().g1, proof.1[0]),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]);

    // Verify gx2
    checks.push(vec![(inputs[1].first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -proof.1[0])]);

    checks
  }
}

#[derive(Debug)]
pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[0].shape() == inputs[1].shape());
    let mut r = ArrayD::zeros(inputs[0].dim());
    azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x * y);
    vec![r]
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0 = &inputs[0].first().unwrap();
    let inp1 = &inputs[1].first().unwrap();
    let out = &outputs[0].first().unwrap();
    let N = inp0.raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = util::msm::<G2Projective>(&srs.X2A, &inp1.poly.coeffs) + srs.Y2P * inp1.r;
    let t = inp0.poly.mul(&inp1.poly).sub(&out.poly).divide_by_vanishing_poly(domain).unwrap().0;

    // Blinding
    let mut rng = StdRng::from_entropy();
    let r = Fr::rand(&mut rng);
    let tx = util::msm::<G1Projective>(&srs.X1A, &t.coeffs) + srs.Y1P * r;
    let C = (inp0.g1 * inp1.r) + (inp1.g1 * inp0.r) + (srs.Y1P * (inp0.r * inp1.r)) - (srs.X1P[0] * out.r) - ((srs.X1P[N] - srs.X1P[0]) * r);
    return (vec![tx, C], vec![gx2], Vec::new());
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = vec![];
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    checks.push(vec![
      (inputs[0].first().unwrap().g1, proof.1[0]),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], (srs.X2A[inputs[0].first().unwrap().len] - srs.X2A[0]).into()),
      (-proof.0[1], srs.Y2A),
    ]);
    // Verify gx2
    checks.push(vec![(inputs[1].first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -proof.1[0])]);
    checks
  }
}
