use ark_bn254::{Fr, G1Affine, G2Affine};
use rand::Rng;
use ndarray::{Array, IxDyn};
use super::{BasicBlock,Data,DataEnc,Tensor};

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  type Setup = (Vec<G1Affine>, Vec<G2Affine>);
  type Proof = (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>);
  fn run(
    _model: &Vec<Tensor<Fr>>,
    inputs: &Vec<Tensor<Fr>>,
  ) -> Vec<Tensor<Fr>> {
    let mut r = Vec::new();
    for i in 0..inputs[0].len() {
      r.push(inputs[0][i] + inputs[1][i]);
    }
    vec![Array::from_shape_vec(IxDyn(inputs[0].shape()), r).unwrap()]
  }
  fn setup(
    _srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
  ) -> Self::Setup {
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(
    _srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: &Self::Setup,
    _model: &Data,
    _inputs: &Vec<Data>,
    _output: &Data,
    _rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) {
    return (Vec::new(), Vec::new(), Vec::new());
  }
  fn verify<R: Rng>(
    _srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    _proof: &Self::Proof,
    _rng: &mut R,
  ) {
    // Verify f(x)+g(x)=h(x)
    let lhs = inputs[0].g1 + inputs[1].g1;
    let rhs = output.g1;
    assert!(lhs == rhs);
  }
}
