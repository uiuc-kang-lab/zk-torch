#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{UniformRand, Zero};
use rand::{rngs::StdRng, SeedableRng};

pub struct SumBasicBlock;
impl BasicBlock for SumBasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>) {
    (vec![], vec![2])
  }
  fn run(&self, _model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![vec![inputs.iter().map(|x| x.iter().sum::<Fr>()).sum::<Fr>()]]
  }
  fn prove(
    &mut self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let l = inputs.len();
    let m = inputs[0].raw.len();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let mut input_raw = vec![Fr::zero(); m];
    let mut input_r = Fr::zero();
    for i in 0..l {
      for j in 0..m {
        input_raw[j] += inputs[i].raw[j];
      }
      input_r += inputs[i].r;
    }
    let input_poly = DensePolynomial {
      coeffs: domain_m.ifft(&input_raw),
    }; //sum poly?

    let mut rng2 = StdRng::from_entropy();
    let zero_div_r = Fr::rand(&mut rng2);
    let zero_div = util::msm::<G1Projective>(&srs.X1A, &input_poly.coeffs[1..]) + srs.Y1P * zero_div_r;
    let C = -srs.X1P[1] * zero_div_r + srs.X1P[0] * (input_r - outputs[0].r * Fr::from(m as u32).inverse().unwrap());
    return (vec![zero_div, C], vec![]);
  }
  fn verify(
    &self,
    srs: &SRS,
    model: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let m = inputs[0].len;
    let [zero_div, C] = proof.0[..] else { panic!("Wrong proof format") };

    let input: G1Projective = inputs.iter().map(|x| x.g1).sum();
    let zero = outputs[0].g1 * Fr::from(m as u32).inverse().unwrap();

    let lhs = Bn254::pairing(input - zero, srs.X2A[0]);
    let rhs = Bn254::pairing(zero_div, srs.X2A[1]) + Bn254::pairing(C, srs.Y2A);
    assert!(lhs == rhs);
  }
}
