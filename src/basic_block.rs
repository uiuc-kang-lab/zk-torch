#![allow(unused_variables)]
use crate::util;
pub use add::AddBasicBlock;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
pub use cq::CQBasicBlock;
pub use cqlin::CQLinBasicBlock;
pub use mul::MulBasicBlock;
use rand::Rng;
pub mod add;
pub mod cq;
pub mod cqlin;
pub mod mul;

pub struct Data {
  pub raw: Vec<Fr>,
  pub dims: Vec<usize>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Affine
}
impl Data {
  pub fn new(srs: (&Vec<G1Affine>, &Vec<G2Affine>), raw: &Vec<Fr>) -> Data {
    let N = (*raw).len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial { coeffs: domain.ifft(raw) };
    let fx: G1Affine = util::msm::<G1Projective>(&srs.0[..N], &f.coeffs).into();
    return Data {
      raw: raw.to_vec(),
      dims: vec![N],
      poly: f,
      g1: fx,
    };
  }

  pub fn new_with_dims(srs: (&Vec<G1Affine>, &Vec<G2Affine>), raw: &Vec<Fr>, dims: Vec<usize>) -> Data {
    let N = (*raw).len();
    println!("N: {}", N);
    assert_eq!(dims.iter().fold(1, |acc, &num| acc * num), N);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial { coeffs: domain.ifft(raw) };
    let fx: G1Affine = util::msm::<G1Projective>(&srs.0[..N], &f.coeffs).into();
    println!("dims {:?}", dims);
    return Data {
      raw: raw.to_vec(),
      dims,
      poly: f,
      g1: fx,
    };
  }
}

#[derive(Debug)]
pub struct DataEnc {
  pub dims: Vec<usize>,
  pub g1: G1Affine,
}
impl DataEnc {
  pub fn new(data: &Data) -> DataEnc {
    return DataEnc {
      dims: data.dims.clone(),
      g1: data.g1,
    };
  }
}
pub trait BasicBlock {
  fn run(model: &Data, inputs: &Vec<Vec<Fr>>) -> Vec<Fr> {
    Vec::new()
  }
  fn setup(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    (Vec::new(), Vec::new())
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Data,
    inputs: &Vec<Data>,
    output: &Data,
    rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    (Vec::new(), Vec::new())
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut R,
  ) {
    ()
  }
}
