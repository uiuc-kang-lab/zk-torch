use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct SubBasicBlock;
impl BasicBlock for SubBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].ndim() <= 1);
    let mut r = ArrayD::zeros(IxDyn(&[std::cmp::max(inputs[0].len(), inputs[1].len())]));
    if inputs[0].len() == 1 && inputs[1].ndim() == 0 {
      // speicial case: [1] - []
      r = inputs[0].map(|x| x - inputs[1].first().unwrap());
    } else if inputs[0].len() == 1 {
      azip!((r in &mut r, &y in inputs[1]) *r = *inputs[0].first().unwrap() - y);
    } else if inputs[1].len() == 1 {
      azip!((r in &mut r, &x in inputs[0]) *r = x - *inputs[1].first().unwrap());
    } else {
      azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x - y);
    }
    vec![r]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let a = &inputs[0].first().unwrap();
    let b = &inputs[1].first().unwrap();
    vec![arr0(Data {
      raw: outputs[0].clone().into_raw_vec(),
      poly: (&a.poly) - (&b.poly),
      g1: a.g1 - b.g1,
      r: a.r - b.r,
    })
    .into_dyn()]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    let c = outputs[0].first().unwrap();
    // Verify f(x)-g(x)=h(x)
    assert!(a.g1 - b.g1 == c.g1);
    vec![]
  }
}
