#![allow(unused_imports)]

use crate::graph::SetupType;
use crate::util;
pub use add::AddBasicBlock;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::UniformRand;
pub use constant::ConstBasicBlock;
pub use cq::CQBasicBlock;
pub use cq2::CQ2BasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use div::DivScalarBasicBlock;
pub use eq::EqBasicBlock;
pub use matmul::MatMulBasicBlock;
pub use max::MaxBasicBlock;
pub use mul::{MulBasicBlock, MulConstBasicBlock, MulScalarBasicBlock};
use ndarray::ArrayD;
pub use ops::{ExpBasicBlock, LogBasicBlock, ReLUBasicBlock, SqrtBasicBlock};
pub use permute::PermuteBasicBlock;
use rand::{rngs::StdRng, SeedableRng};
pub use rope::RoPEBasicBlock;
pub use squeeze::{SqueezeBasicBlock, UnsqueezeBasicBlock};
pub use sub::SubBasicBlock;
pub use sum::SumBasicBlock;

pub mod add;
pub mod constant;
pub mod cq;
pub mod cq2;
pub mod cqlin;
pub mod div;
pub mod eq;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod ops;
pub mod permute;
pub mod rope;
pub mod squeeze;
pub mod sub;
pub mod sum;

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

pub struct Data {
  pub raw: Vec<Fr>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Projective,
  pub r: Fr,
}

impl Data {
  pub fn new(srs: &SRS, raw: &[Fr]) -> Data {
    let N = raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial { coeffs: domain.ifft(&raw) };
    let fx = util::msm::<G1Projective>(&srs.X1A, &f.coeffs);
    let mut rng = StdRng::from_entropy();
    return Data {
      raw: raw.to_vec(),
      poly: f,
      g1: fx,
      r: Fr::rand(&mut rng),
    };
  }
}

pub struct DataEnc {
  pub len: usize,
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

#[derive(Debug)]
pub enum BasicBlockType {
  Add,
  ChangeSF,
  Constant,
  CQ,
  CQ2,
  CQLin,
  Div,
  Eq,
  Exp,
  Log,
  MatMul,
  Max,
  Mul,
  MulConst,
  MulScalar,
  Permute,
  ReLU,
  RoPE,
  Sqrt,
  Squeeze,
  Sub,
  Sum,
  Unsqueeze,
}

pub trait BasicBlock: std::fmt::Debug {
  fn block_type(&self) -> BasicBlockType {
    BasicBlockType::Constant
  }

  fn name(&self) -> String {
    format!("{self:?}")
  }

  fn weights_name(&self) -> Result<String, String> {
    Err(format!("{:?} does not use weights.", self.block_type()))
  }

  fn setup(&self, _srs: &SRS, _model: &ArrayD<Fr>) -> SetupType {
    SetupType::None
  }

  fn run(&self, _weights: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![]
  }

  fn prove(
    &mut self,
    _srs: &SRS,
    _setup: &SetupType,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    (Vec::new(), Vec::new())
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut StdRng,
  ) {
    ()
  }
}

/// Hash, PartialEq, Eq are implemented for dyn BasicBlock trait objects
/// so that BasicBlock can be used in HashMaps, which is useful for tracking
/// the usage of BasicBlocks in a graph. This prevents redundant setups for
/// the same BasicBlock (e.g., we use CQ2 for ReLU and Log, but we don't
/// want to setup CQ2 twice for ReLU)
impl std::hash::Hash for dyn BasicBlock {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.name().hash(state);
  }
}

impl std::cmp::PartialEq for dyn BasicBlock {
  fn eq(&self, other: &Self) -> bool {
    self.name() == other.name()
  }
}

impl std::cmp::Eq for dyn BasicBlock {}
