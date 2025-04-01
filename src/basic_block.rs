#![allow(unused_imports)]
use crate::util::{self, ark_de, ark_se, AccHolder, AccProofLayout};
pub use add::{AddAlongAxisBasicBlock, AddBasicBlock, MultipleAddBasicBlock};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, Zero};
pub use bool_check::BooleanCheckBasicBlock;
pub use clip::ClipBasicBlock;
pub use concat::{ConcatBasicBlock, ConcatLastDimBasicBlock};
pub use constant::{Const2BasicBlock, ConstBasicBlock, ConstOfShapeBasicBlock};
pub use conv::{Conv2DAddBasicBlock, Conv3DAddBasicBlock, Conv3DTransposeBasicBlock};
pub use copy_constraint::CopyConstraintBasicBlock;
pub use cq::CQBasicBlock;
pub use cq2::CQ2BasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use div::{DivConstBasicBlock, DivConstProofBasicBlock, DivScalarBasicBlock, ModConstBasicBlock};
use downcast_rs::impl_downcast;
pub use eq::{ElementwiseEqBasicBlock, EqBasicBlock};
pub use id::IdBasicBlock;
pub use less::{GreaterBasicBlock, LessBasicBlock};
pub use matmul::MatMulBasicBlock;
pub use max::{MaxBasicBlock, MaxProofBasicBlock};
pub use mul::{MulBasicBlock, MulConstBasicBlock, MulScalarBasicBlock};
use ndarray::{ArrayD, IxDyn};
pub use one_to_one::OneToOneBasicBlock;
pub use ops::*;
pub use ordered::OrderedBasicBlock;
pub use permute::PermuteBasicBlock;
use rand::{rngs::StdRng, SeedableRng};
pub use range::RangeConstBasicBlock;
use rayon::prelude::*;
pub use repeater::RepeaterBasicBlock;
pub use reshape::ReshapeBasicBlock;
pub use rope::RoPEBasicBlock;
use serde::{Deserialize, Serialize};
pub use sort::SortBasicBlock;
pub use split::SplitBasicBlock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
pub use sub::SubBasicBlock;
pub use sum::SumBasicBlock;
pub use transpose::TransposeBasicBlock;

pub mod add;
pub mod bool_check;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod conv;
pub mod copy_constraint;
pub mod cq;
pub mod cq2;
pub mod cqlin;
pub mod div;
pub mod eq;
pub mod id;
pub mod less;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod one_to_one;
pub mod ops;
pub mod ordered;
pub mod permute;
pub mod range;
pub mod repeater;
pub mod reshape;
pub mod rope;
pub mod sort;
pub mod split;
pub mod sub;
pub mod sum;
pub mod transpose;

pub struct SRS {
  pub X1A: Vec<G1Affine>,
  pub X2A: Vec<G2Affine>,
  pub X1P: Vec<G1Projective>,
  pub X2P: Vec<G2Projective>,
  pub Y1A: G1Affine,
  pub Y2A: G2Affine,
  pub Y1P: G1Projective,
  pub Y2P: G2Projective,
}

// During proofs and verifications, a cache is used to prevent recomputation.
// These are the types of the elements in the cache.
pub enum CacheValues {
  CQTableDict(HashMap<Fr, usize>),
  CQ2TableDict(HashMap<(Fr, Fr), usize>),
  RLCRandom(Fr),
  Data(Data),
  G2(G2Affine),
}

// The cache is wrapped in Arc<Mutex<>> to allow multiple threads within the same role (either prover or verifier) to access it.
// Arc (Atomic Reference Counting) enables safe sharing of the cache between threads,
// while Mutex ensures that only one thread can write to the cache at a time, preventing race conditions.
// Note: Each prover and verifier maintains its own separate cache. There is no cache sharing between the prover and verifier.
pub type ProveVerifyCache = Arc<Mutex<HashMap<String, CacheValues>>>;

pub type PairingCheck = Vec<(G1Affine, G2Affine)>;

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Data {
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub raw: Vec<Fr>,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub poly: DensePolynomial<Fr>,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub g1: G1Projective,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub r: Fr,
}

impl Data {
  pub fn new(srs: &SRS, raw: &[Fr]) -> Data {
    let N = raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial::from_coefficients_vec(domain.ifft(&raw));
    let fx = if f.is_zero() {
      G1Projective::zero()
    } else {
      util::msm(&srs.X1A, &f.coeffs)
    };
    let mut rng = StdRng::from_entropy();
    return Data {
      raw: raw.to_vec(),
      poly: f,
      g1: fx,
      r: Fr::rand(&mut rng),
    };
  }
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct DataEnc {
  pub len: usize,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub g1: G1Affine,
}

impl DataEnc {
  pub fn new(srs: &SRS, data: &Data) -> DataEnc {
    return DataEnc {
      len: data.raw.len(),
      g1: (data.g1 + srs.Y1P * data.r).into(),
    };
  }
}

pub trait BasicBlock: std::fmt::Debug + Send + Sync + downcast_rs::Downcast {
  fn genModel(&self) -> ArrayD<Fr> {
    ArrayD::zeros(IxDyn(&[0]))
  }

  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    Ok(vec![])
  }

  // This function encodes the outputs of the BasicBlock into Data objects.
  // It defaults to running Data::new() on the last dimension of the outputs which runs an FFT and an MSM.
  // But for certain basic blocks such as add and reshape, this can be done much faster, and it should be overriden in these cases.
  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    util::vec_iter(outputs).map(|x| util::convert_to_data(srs, x)).collect()
  }

  // The subsequent setup/prove/verify functions run on encoded Data objects (vector commitments).
  // This reduces computation because the Data objects can be encoded once at the beginning and then reused for these functions.
  fn setup(&self, _srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    #[cfg(feature = "mock_prove")]
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    (Vec::new(), Vec::new(), Vec::new())
  }

  fn prove(
    &self,
    _srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    (Vec::new(), Vec::new(), Vec::new())
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    vec![]
  }

  // This function performs folding for the rest of the blocks in the computation
  fn acc_prove(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
    _proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
  }

  // This function cleans the blinding terms in accumulators for the verifier to do acc_verify
  fn acc_clean(
    &self,
    _srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
  ) -> (
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>),
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) {
    (
      (
        proof.0.iter().map(|x| (*x).into()).collect(),
        proof.1.iter().map(|x| (*x).into()).collect(),
        proof.2.iter().map(|x| *x).collect(),
      ),
      (
        acc_proof.0.iter().map(|x| (*x).into()).collect(),
        acc_proof.1.iter().map(|x| (*x).into()).collect(),
        acc_proof.2.iter().map(|x| *x).collect(),
        acc_proof.3.iter().map(|x| *x).collect(),
      ),
    )
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    _prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    _acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    None
  }

  // This function is used to clean the errs in the final accumulator proof to calculate the proof size correctly.
  fn acc_finalize(
    &self,
    _srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    (
      acc_proof.0.iter().map(|x| *x).collect(),
      acc_proof.1.iter().map(|x| *x).collect(),
      acc_proof.2.iter().map(|x| *x).collect(),
      acc_proof.3.iter().map(|x| *x).collect(),
    )
  }

  fn acc_decide(
    &self,
    _srs: &SRS,
    _acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    vec![]
  }
}

#[derive(Debug)]
pub struct BasicBlockForTest;
impl BasicBlock for BasicBlockForTest {}
impl AccProofLayout for BasicBlockForTest {
  fn acc_g1_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn acc_fr_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn prover_proof_to_acc(&self, _proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    AccHolder {
      acc_g1: vec![],
      acc_g2: vec![],
      acc_fr: vec![],
      mu: Fr::zero(),
      errs: vec![],
      acc_errs: vec![],
    }
  }
  fn verifier_proof_to_acc(&self, _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    AccHolder {
      acc_g1: vec![],
      acc_g2: vec![],
      acc_fr: vec![],
      mu: Fr::zero(),
      errs: vec![],
      acc_errs: vec![],
    }
  }
  fn mira_prove(
    &self,
    _srs: &SRS,
    _acc_1: AccHolder<G1Projective, G2Projective>,
    _acc_2: AccHolder<G1Projective, G2Projective>,
    _rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    AccHolder {
      acc_g1: vec![],
      acc_g2: vec![],
      acc_fr: vec![],
      mu: Fr::zero(),
      errs: vec![],
      acc_errs: vec![],
    }
  }
}

impl_downcast!(BasicBlock);
