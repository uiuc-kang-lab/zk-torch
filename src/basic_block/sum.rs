#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{UniformRand, Zero};
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct SumBasicBlock;
impl BasicBlock for SumBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    vec![arr1(&[inputs[0].iter().sum::<Fr>()]).into_dyn()]
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
    let input = inputs[0].first().unwrap();
    let m = input.raw.len();

    let mut rng2 = StdRng::from_entropy();
    let zero_div_r = Fr::rand(&mut rng2);
    let zero_div = if input.poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &input.poly.coeffs[1..])
    } + srs.Y1P * zero_div_r;
    let C = -srs.X1P[1] * zero_div_r + srs.X1P[0] * (input.r - outputs[0].first().unwrap().r * Fr::from(m as u32).inverse().unwrap());
    return (vec![zero_div, C], vec![], Vec::new());
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
    let input = inputs[0].first().unwrap();
    let m = input.len;
    let [zero_div, C] = proof.0[..] else { panic!("Wrong proof format") };

    let zero = outputs[0].first().unwrap().g1 * Fr::from(m as u32).inverse().unwrap();
    let input_g1: G1Projective = input.g1.into();

    vec![vec![((input_g1 - zero).into(), srs.X2A[0]), (-zero_div, srs.X2A[1]), (-C, srs.Y2A)]]
  }
}
