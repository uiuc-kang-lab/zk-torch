#![allow(unused_imports)]
use crate::util::{self, ark_de, ark_se};
pub use add::AddBasicBlock;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, Zero};
pub use clip::ClipBasicBlock;
pub use concat::ConcatBasicBlock;
pub use constant::{Const2BasicBlock, ConstBasicBlock, ConstOfShapeBasicBlock};
pub use copy_constraint::CopyConstraintBasicBlock;
pub use cq::CQBasicBlock;
pub use cq2::CQ2BasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use div::{DivConstBasicBlock, DivScalarBasicBlock};
pub use eq::{ElementwiseEqBasicBlock, EqBasicBlock};
pub use id::IdBasicBlock;
pub use matmul::MatMulBasicBlock;
pub use max::{MaxBasicBlock, MaxProofBasicBlock};
pub use mul::{MulBasicBlock, MulConstBasicBlock, MulScalarBasicBlock};
use ndarray::{ArrayD, IxDyn};
pub use ops::*;
pub use permute::PermuteBasicBlock;
use rand::{rngs::StdRng, SeedableRng};
pub use range::RangeConstBasicBlock;
use rayon::prelude::*;
pub use repeater::RepeaterBasicBlock;
pub use reshape::ReshapeBasicBlock;
pub use rope::RoPEBasicBlock;
use serde::{Deserialize, Serialize};
pub use split::SplitBasicBlock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use sub::SubBasicBlock;
pub use sum::SumBasicBlock;
pub use transpose::TransposeBasicBlock;
pub mod add;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod copy_constraint;
pub mod cq;
pub mod cq2;
pub mod cqlin;
pub mod div;
pub mod eq;
pub mod id;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod ops;
pub mod permute;
pub mod range;
pub mod repeater;
pub mod reshape;
pub mod rope;
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

pub trait BasicBlock: std::fmt::Debug + Send + Sync {
  fn genModel(&self) -> ArrayD<Fr> {
    ArrayD::zeros(IxDyn(&[0]))
  }

  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![]
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
}
