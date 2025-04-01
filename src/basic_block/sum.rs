#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_proof_to_acc, acc_to_acc_proof, AccHolder, AccProofLayout};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{One, UniformRand, Zero};
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

impl AccProofLayout for SumBasicBlock {
  fn acc_g1_num(&self, _is_prover: bool) -> usize {
    4
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn acc_fr_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::one(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    }
  }
  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    AccHolder {
      acc_g1: proof.0.clone(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::one(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    }
  }
  fn mira_prove(
    &self,
    _srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_1.acc_g1.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);
    AccHolder {
      acc_g1: acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: acc_1.mu + acc_gamma * acc_2.mu,
      errs: Vec::new(),
      acc_errs: Vec::new(),
    }
  }
  fn mira_verify(
    &self,
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    let mut result = true;
    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_1.acc_g1.serialize_uncompressed(&mut bytes).unwrap();
    acc_2.acc_g1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    acc_2.acc_g1.iter().zip(acc_1.acc_g1.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *x * acc_gamma + *y;
      result &= new_acc.acc_g1[i] == z;
    });
    result &= new_acc.mu == acc_1.mu + acc_gamma * acc_2.mu;

    Some(result)
  }
}

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

  #[cfg(not(feature = "fold"))]
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

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let proof = self.prover_proof_to_acc(proof);
    if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
      return acc_to_acc_proof(proof);
    }
    let acc_proof = acc_proof_to_acc(self, acc_proof, true);
    acc_to_acc_proof(self.mira_prove(srs, acc_proof, proof, rng))
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let mut result = inputs[0].first().unwrap().g1 == proof.0[2] && outputs[0].first().unwrap().g1 == proof.0[3];
    if prev_acc_proof.2.len() == 0 && acc_proof.2[0].is_one() {
      return Some(result);
    }
    let proof = self.verifier_proof_to_acc(proof);
    let acc_proof = acc_proof_to_acc(self, acc_proof, false);
    let prev_acc_proof = acc_proof_to_acc(self, prev_acc_proof, false);
    result &= self.mira_verify(prev_acc_proof, proof, acc_proof, rng).unwrap();
    Some(result)
  }

  fn acc_decide(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let [zero_div, C, inp, out] = acc_proof.0[..] else {
      panic!("Wrong proof format")
    };
    let zero = out * Fr::from(self.len as u32).inverse().unwrap();
    vec![(
      vec![((-zero + inp).into(), srs.X2A[0]), (-zero_div, srs.X2A[1]), (-C, srs.Y2A)],
      PairingOutput::<Bn<ark_bn254::Config>>::zero(),
    )]
  }
}
