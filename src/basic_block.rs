#![allow(unused_variables)]
use crate::util;
pub use add::AddBasicBlock;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::UniformRand;
pub use constant::ConstBasicBlock;
pub use cq::CQBasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use mul::MulBasicBlock;
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
pub use relu::ReLUBasicBlock;
pub mod add;
pub mod constant;
pub mod cq;
pub mod cqlin;
pub mod mul;
pub mod relu;

pub struct Data {
  pub raw: ArrayD<Fr>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Affine,
  pub r: Fr,
}
impl Data {
  pub fn new(srs: (&Vec<G1Affine>, &Vec<G2Affine>), raw: &ArrayD<Fr>) -> Data {
    let N = raw.len();
    let vec = raw.clone().into_raw_vec();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial { coeffs: domain.ifft(&vec) };
    let fx: G1Affine = util::msm::<G1Projective>(&srs.0[..N], &f.coeffs).into();
    let mut rng = StdRng::from_entropy();
    return Data {
      raw: raw.clone(),
      poly: f,
      g1: fx,
      r: Fr::rand(&mut rng),
    };
  }
}
pub struct DataEnc {
  pub len: usize,
  pub shape: Vec<usize>,
  pub g1: G1Affine,
}
impl DataEnc {
  pub fn new(srs: (&Vec<G1Affine>, &Vec<G2Affine>), data: &Data) -> DataEnc {
    return DataEnc {
      len: data.raw.len(),
      shape: data.raw.shape().to_vec(),
      g1: (data.g1 + srs.0[srs.1.len() - 1] * data.r).into(),
    };
  }
}
pub trait BasicBlock {
  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> ArrayD<Fr> {
    ArrayD::zeros(vec![])
  }
  fn setup(&self, srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    (Vec::new(), Vec::new())
  }
  fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Data,
    inputs: &Vec<&Data>,
    output: &Data,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    (Vec::new(), Vec::new())
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
    ()
  }
}
