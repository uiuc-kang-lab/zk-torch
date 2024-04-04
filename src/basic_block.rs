#![allow(unused_variables)]
#![allow(unused_imports)]
use crate::util;
pub use add::AddBasicBlock;
pub use alternate::{CombineBasicBlock, SplitBasicBlock};
pub use alternating::AlternatingBasicBlock;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::UniformRand;
pub use concat::ConcatBasicBlock;
pub use constant::ConstBasicBlock;
pub use cq::CQBasicBlock;
pub use cq2::CQ2BasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use div::{DivConstBasicBlock, DivScalarBasicBlock};
pub use eq::EqBasicBlock;
pub use exp::ExpBasicBlock;
pub use log::LogBasicBlock;
pub use matmul::MatMulBasicBlock;
pub use max::MaxBasicBlock;
pub use mul::{MulBasicBlock, MulConstBasicBlock, MulScalarBasicBlock};
use rand::{rngs::StdRng, SeedableRng};
pub use relu::ReLUBasicBlock;
pub use rope::RoPEBasicBlock;
pub use sqrt::SqrtBasicBlock;
pub use sub::SubBasicBlock;
pub use sum::SumBasicBlock;
pub use transpose::TransposeBasicBlock;
pub mod add;
pub mod alternate;
pub mod alternating;
pub mod concat;
pub mod constant;
pub mod cq;
pub mod cq2;
pub mod cqlin;
pub mod div;
pub mod eq;
pub mod exp;
pub mod log;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod relu;
pub mod rope;
pub mod sqrt;
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
pub struct Data {
  pub raw: Vec<Fr>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Projective,
  pub r: Fr,
}
impl Data {
  pub fn new(srs: &SRS, raw: &Vec<Fr>) -> Data {
    let N = raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial { coeffs: domain.ifft(&raw) };
    let fx = util::msm::<G1Projective>(&srs.X1A, &f.coeffs);
    let mut rng = StdRng::from_entropy();
    return Data {
      raw: raw.clone(),
      poly: f,
      g1: fx,
      r: Fr::rand(&mut rng),
    };
  }
}
impl Clone for Data {
  fn clone(&self) -> Data {
    Data {
      raw: self.raw.clone(),
      poly: self.poly.clone(),
      g1: self.g1.clone(),
      r: self.r.clone(),
    }
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
impl Clone for DataEnc {
  fn clone(&self) -> DataEnc {
    DataEnc {
      len: self.len.clone(),
      g1: self.g1.clone(),
    }
  }
}
pub trait BasicBlock {
  fn get_dims(&self) -> (Vec<usize>, Vec<usize>);
  fn run(&self, model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) -> Vec<Vec<Fr>> {
    vec![]
  }
  fn setup(&self, srs: &SRS, model: &Vec<&Data>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    (Vec::new(), Vec::new())
  }
  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    (Vec::new(), Vec::new())
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
    ()
  }
}
