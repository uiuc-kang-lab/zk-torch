#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{One, UniformRand, Zero};
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct SumBasicBlock {
  pub len: usize,
}
impl BasicBlock for SumBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    Ok(vec![arr1(&[inputs[0].iter().sum::<Fr>()]).into_dyn()])
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
    assert!(m == self.len);

    let mut rng2 = StdRng::from_entropy();
    let zero_div_r = Fr::rand(&mut rng2);
    let zero_div = if input.poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &input.poly.coeffs[1..])
    } + srs.Y1P * zero_div_r;
    let C = -srs.X1P[1] * zero_div_r + srs.X1P[0] * (input.r - outputs[0].first().unwrap().r * Fr::from(m as u32).inverse().unwrap());

    let mut proof = vec![zero_div, C];
    #[cfg(feature = "fold")]
    {
      let inp = inputs[0].first().unwrap();
      let out = outputs[0].first().unwrap();
      let mut additional_g1_for_acc = vec![inp.g1 + srs.Y1P * inp.r, out.g1 + srs.Y1P * out.r];
      proof.append(&mut additional_g1_for_acc);
    }

    return (proof, vec![], Vec::new());
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
    assert!(m == self.len);
    let [zero_div, C] = proof.0[..] else { panic!("Wrong proof format") };

    let zero = outputs[0].first().unwrap().g1 * Fr::from(m as u32).inverse().unwrap();
    let input_g1: G1Projective = input.g1.into();

    vec![vec![((input_g1 - zero).into(), srs.X2A[0]), (-zero_div, srs.X2A[1]), (-C, srs.Y2A)]]
  }

  fn acc_init(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);
    // mu
    acc_proof.2.push(Fr::one());
    acc_proof
  }

  fn acc_prove(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_proof.0.serialize_uncompressed(&mut bytes).unwrap();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let new_acc_proof_g1 = proof.0.iter().zip(acc_proof.0.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    let new_acc_proof_mu = acc_proof.2[0] + acc_gamma;
    (new_acc_proof_g1, Vec::new(), vec![new_acc_proof_mu])
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let mut result = inputs[0].first().unwrap().g1 == proof.0[2] && outputs[0].first().unwrap().g1 == proof.0[3];
    if prev_acc_proof.2.len() == 0 && acc_proof.2[0].is_one() {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_proof.0.serialize_uncompressed(&mut bytes).unwrap();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    proof.0.iter().zip(prev_acc_proof.0.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *x * acc_gamma + *y;
      result &= acc_proof.0[i] == z;
    });
    result &= acc_proof.2[0] == prev_acc_proof.2[0] + acc_gamma;

    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let [zero_div, C, inp, out] = acc_proof.0[..] else {
      panic!("Wrong proof format")
    };
    let zero = out * Fr::from(self.len as u32).inverse().unwrap();
    vec![vec![((-zero + inp).into(), srs.X2A[0]), (-zero_div, srs.X2A[1]), (-C, srs.Y2A)]]
  }
}
