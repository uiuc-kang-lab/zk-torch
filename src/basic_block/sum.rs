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
    assert!(inputs.len() == 1 && inputs[0].ndim() == 2);
    vec![arr1(&[inputs[0].iter().sum::<Fr>()]).into_dyn()]
  }

  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let input = inputs[0];
    let l = input.len();
    let m = input[0].raw.len();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let mut input_raw = vec![Fr::zero(); m];
    let mut input_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        input_raw[j] += input[i].raw[j];
      }
      input_r += input[i].r;
    }
    let input_poly = DensePolynomial::from_coefficients_vec(domain_m.ifft(&input_raw)); // sum poly?

    let mut rng2 = StdRng::from_entropy();
    let zero_div_r = Fr::rand(&mut rng2);
    let zero_div = if input_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &input_poly.coeffs[1..])
    } + srs.Y1P * zero_div_r;
    let C = -srs.X1P[1] * zero_div_r + srs.X1P[0] * (input_r - outputs[0][0].r * Fr::from(m as u32).inverse().unwrap());
    return (vec![zero_div, C], vec![]);
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let m = inputs[0][0].len;
    let [zero_div, C] = proof.0[..] else { panic!("Wrong proof format") };

    let input: G1Projective = inputs[0].iter().map(|x| x.g1).sum();
    let zero = outputs[0][0].g1 * Fr::from(m as u32).inverse().unwrap();

    vec![vec![((input - zero).into(), srs.X2A[0]), (-zero_div, srs.X2A[1]), (-C, srs.Y2A)]]
  }
}
