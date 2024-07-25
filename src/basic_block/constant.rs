use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, ArrayD, IxDyn};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct ConstBasicBlock;
impl BasicBlock for ConstBasicBlock {
  fn run(&self, model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![model.clone()]
  }

  fn encodeOutputs(&self, _srs: &SRS, model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    vec![model.clone()]
  }

  fn verify(
    &self,
    _srs: &SRS,
    model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    assert!(model == outputs[0]);

    vec![]
  }
}

#[derive(Debug)]
pub struct Const2BasicBlock {
  pub c: ArrayD<Fr>,
}

impl BasicBlock for Const2BasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![self.c.clone()]
  }
}

// ConstOfShapeBasicBlock is a basic block that creates a constant tensor of a given shape and value.
// It requires no proving since the constant value is known as a public input.
#[derive(Debug)]
pub struct ConstOfShapeBasicBlock {
  pub c: Fr,
  pub shape: Vec<usize>,
}
impl BasicBlock for ConstOfShapeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, _inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    vec![ArrayD::from_elem(IxDyn(&self.shape), self.c)]
  }
}
