use crate::basic_block::*;
use crate::{convert_to_data, ptau, util};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::univariate::DensePolynomial;
use ark_std::UniformRand;
use ark_std::Zero;
use ndarray::{arr0, concatenate, s, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

fn testBasicBlock<BB: BasicBlock>(mut basic_block: BB, srs: &SRS, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let outputs = basic_block.run(model, inputs);
  let outputs: Vec<&ArrayD<Fr>> = outputs.iter().map(|x| x).collect();
  let model = convert_to_data(srs, model);
  let setup = basic_block.setup(srs, &model);
  let setup: (Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>) = (
    setup.0.iter().map(|y| (*y).into()).collect(),
    setup.1.iter().map(|y| (*y).into()).collect(),
    setup.2.iter().map(|y| (y.clone())).collect(),
  );
  let setup = (&setup.0, &setup.1, &setup.2);
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<Data>> = basic_block.encodeOutputs(srs, &model, &inputs, &outputs);
  let outputs: Vec<&ArrayD<Data>> = outputs.iter().map(|output| output).collect();
  let mut rng2 = rng.clone();
  let mut cache = HashMap::new();
  let proof = basic_block.prove(srs, setup, &model, &inputs, &outputs, &mut rng, &mut cache);
  let proof: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (
    proof.0.iter().map(|y| (*y).into()).collect(),
    proof.1.iter().map(|y| (*y).into()).collect(),
    proof.2.iter().map(|y| (*y)).collect(),
  );
  let proof = (&proof.0, &proof.1, &proof.2);
  let model = model.map(|x| DataEnc::new(srs, x));
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<DataEnc>> = outputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<&ArrayD<DataEnc>> = outputs.iter().map(|output| output).collect();
  let mut cache = HashMap::new();
  let pairings = basic_block.verify(srs, &model, &inputs, &outputs, proof, &mut rng2, &mut cache);
  let pairings = pairings.iter().map(|x| x).collect();
  let pairings = util::combine_pairing_checks(&pairings);
  assert_eq!(Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()), PairingOutput::zero());
}

#[test]
fn testBasicBlocks() {
  let srs = &ptau::load_file("challenge", 7, 7);
  let mut rng = StdRng::from_entropy();
  let N: usize = 1 << 6;
  let a = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let b = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let empty = ArrayD::zeros(IxDyn(&[0]));
  testBasicBlock(MulBasicBlock {}, srs, &empty, &vec![&a, &b]);
}
