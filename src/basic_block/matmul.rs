#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use ndarray::{arr1, ArrayD};
use rand::rngs::StdRng;

pub struct MatMulBasicBlock;
impl BasicBlock for MatMulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> ArrayD<Fr> {
    let m = inputs.len() - 1;
    let n = inputs[0].shape()[0];
    let mut r = ArrayD::zeros(vec![m]);
    for i in 0..m {
      for j in 0..n {
        r[i] += inputs[1 + i][j] * inputs[0][j];
      }
    }
    return r;
  }
  fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    output: &Data,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let m = inputs.len() - 1;
    let n = inputs[0].raw.shape()[0];
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let alpha = Fr::rand(rng);

    //calculate powers
    let mut pow: Vec<Fr> = vec![Fr::one(); m];
    for i in 0..m - 1 {
      pow[i + 1] = pow[i] * alpha;
    }
    let pow_poly = DensePolynomial { coeffs: domain_m.ifft(&pow) };

    //calculate flat
    let mut flat = vec![Fr::zero(); n];
    for i in 0..m {
      for j in 0..n {
        flat[j] += inputs[1 + i].raw[j] * pow[i];
      }
    }
    let flat_poly = DensePolynomial {
      coeffs: domain_n.ifft(&flat),
    };

    let A: Vec<Fr> = (0..n).map(|i| flat[i] * inputs[0].raw[i]).collect();
    let A = Data::new(srs, &arr1(&A).into_dyn());
    //A pointwise mul proof
    let gx2A = util::msm::<G2Projective>(&srs.1, &inputs[0].poly.coeffs);
    let tA = flat_poly.mul(&inputs[0].poly).sub(&A.poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let txA = util::msm::<G1Projective>(&srs.0, &tA.coeffs);
    //A zero proof
    let A_zero = srs.0[0] * (Fr::from(n as u32).inverse().unwrap() * A.raw.iter().sum::<Fr>());
    let A_zero_div = util::msm::<G1Projective>(&srs.0, &A.poly.coeffs[1..]);

    let B: Vec<Fr> = (0..m).map(|i| output.raw[i] * pow[i]).collect();
    let B = Data::new(srs, &arr1(&B).into_dyn());
    //B pointwise mul proof
    let tB = output.poly.mul(&pow_poly).sub(&B.poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let txB = util::msm::<G1Projective>(&srs.0, &tB.coeffs);
    //B zero proof
    let B_zero_div = util::msm::<G1Projective>(&srs.0, &B.poly.coeffs[1..]);

    let proof1: Vec<G1Projective> = vec![A.g1.into(), txA, A_zero, A_zero_div, B.g1.into(), txB, B_zero_div];
    let proof2: Vec<G2Projective> = vec![gx2A];
    let proof1: Vec<G1Affine> = proof1.iter().map(|x| (*x).into()).collect();
    let proof2: Vec<G2Affine> = proof2.iter().map(|x| (*x).into()).collect();
    return (proof1, proof2);
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let m = inputs.len() - 1;
    let n = inputs[0].shape[0];
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let alpha = Fr::rand(rng);

    //calculate powers
    let mut pow: Vec<Fr> = vec![Fr::one(); m]; //calculated directly
    for i in 0..m - 1 {
      pow[i + 1] = pow[i] * alpha;
    }
    let pow_poly = DensePolynomial { coeffs: domain_m.ifft(&pow) };
    let pow_x2 = util::msm::<G2Projective>(&srs.1, &pow_poly.coeffs);

    //calculate flat
    let mut flat_x = G1Projective::zero();
    for i in 0..m {
      flat_x += inputs[1 + i].g1 * pow[i];
    }

    //check A=flat_x * inputs[0]
    let lhs = Bn254::pairing(flat_x, proof.1[0]) - Bn254::pairing(proof.0[0], srs.1[0]);
    let rhs = Bn254::pairing(proof.0[1], srs.1[n] - srs.1[0]);
    assert!(lhs == rhs);
    let lhs = Bn254::pairing(inputs[0].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], proof.1[0]);
    assert!(lhs == rhs);

    //check A(0)
    let lhs = Bn254::pairing(proof.0[0] - proof.0[2], srs.1[0]);
    let rhs = Bn254::pairing(proof.0[3], srs.1[1]);
    assert!(lhs == rhs);

    //check B=output * pow
    let lhs = Bn254::pairing(output.g1, pow_x2) - Bn254::pairing(proof.0[4], srs.1[0]);
    let rhs = Bn254::pairing(proof.0[5], srs.1[m] - srs.1[0]);
    assert!(lhs == rhs);

    //check B(0)
    let B_zero: G1Affine = (proof.0[2] * (Fr::from(n as u32) * Fr::from(m as u32).inverse().unwrap())).into();
    let lhs = Bn254::pairing(proof.0[4] - B_zero, srs.1[0]);
    let rhs = Bn254::pairing(proof.0[6], srs.1[1]);
    assert!(lhs == rhs);
  }
}
