use ark_poly::univariate::DensePolynomial;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine};
use rand::Rng;
use ndarray::{Array, IxDyn};
use crate::util;
pub use cq::CQBasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use mul::MulBasicBlock;
pub use add::AddBasicBlock;
pub use rope::RopeBasicBlock;
pub use transpose::TransposeBasicBlock;
pub use matmult::MatMultBasicBlock;
pub use bridge::BridgeBasicBlock;
pub mod cq;
pub mod cqlin;
pub mod mul;
pub mod add;
pub mod rope;
pub mod transpose;
pub mod matmult;
pub mod bridge;

pub struct Data{
  pub raw : Tensor<Fr>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Affine
}
impl Data{
  pub fn new(srs:(&Vec<G1Affine>,&Vec<G2Affine>), raw:&Tensor<Fr>) -> Data{
    let raw_vec = raw.clone().into_raw_vec();
    let N = (*raw_vec).len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial{coeffs: domain.ifft(&raw_vec)};
    let fx: G1Affine = util::msm::<G1Projective>(&srs.0[..N], &f.coeffs).into();
    return Data{raw: raw.clone(), poly: f, g1: fx};
  }
}
pub struct DataEnc{
  pub len : usize,
  pub g1: G1Affine,
  pub shape: Vec<usize>,
}
impl DataEnc{
  pub fn new(data:&Data) -> DataEnc{
    return DataEnc{len: data.raw.len(), g1: data.g1, shape: data.raw.shape().to_owned()};
  }
}

pub type Tensor<F> = Array<F, IxDyn>;
pub trait BasicBlock{
  type Setup;
  type Proof;
  fn run(model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
         Vec<Tensor<Fr>>;
  fn setup(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           model: &Data) -> Self::Setup;
          //(Vec<G1Affine>,Vec<G2Affine>);
  fn prove<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   setup: &Self::Setup,
                   model: &Data,
                   inputs: &Vec<Data>,
                   output: &Data,
                   rng: &mut R) -> Self::Proof;
                  //(Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    output: &DataEnc,
                    proof: &Self::Proof,
                    rng: &mut R);
}


