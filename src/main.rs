#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use basic_block::*;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
mod basic_block;
mod ptau;
mod util;

fn test_basic_block<BB: BasicBlock>(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Vec<Fr>, inputs: &Vec<Vec<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let model = Data::new(srs, model);
  let output = BB::run(&model, inputs);
  let setup = BB::setup(srs, &model);
  let inputs = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs, (&(setup.0), &(setup.1)), &model, &inputs, &output, &mut rng);
  let model = DataEnc::new(&model);
  let inputs = inputs.iter().map(|x| DataEnc::new(x)).collect();
  let output = DataEnc::new(&output);
  BB::verify(srs, &model, &inputs, &output, (&(proof.0), &(proof.1)), &mut rng2);
}

fn test_basic_block_with_dims<BB: BasicBlock>(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Vec<Fr>, inputs: &Vec<Vec<Fr>>, dims: &Vec<usize>) {
  let mut rng = StdRng::from_entropy();
  let model = Data::new_with_dims(srs, model, dims.to_vec());
  let output = BB::run(&model, inputs);
  let setup = BB::setup(srs, &model);
  let inputs = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs, (&(setup.0), &(setup.1)), &model, &inputs, &output, &mut rng);
  let model = DataEnc::new(&model);
  let inputs = inputs.iter().map(|x| DataEnc::new(x)).collect();
  let output = DataEnc::new(&output);
  BB::verify(srs, &model, &inputs, &output, (&(proof.0), &(proof.1)), &mut rng2);
}
fn main() {
  let srs = ptau::load_file("challenge", 7);
  let srs = (&srs.0, &srs.1);
  const N: usize = 1 << 6;
  const n: usize = 1 << 3;
  const m: usize = 1 << 2;
  let a: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let b: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  test_basic_block::<AddBasicBlock>(srs, &Vec::new(), &vec![a.clone(), b.clone()]);
  test_basic_block::<MulBasicBlock>(srs, &Vec::new(), &vec![a.clone(), b.clone()]);
  test_basic_block::<CQBasicBlock>(srs, &a, &vec![a[..n].to_vec()]);
  test_basic_block_with_dims::<CQLinBasicBlock>(srs, &a, &vec![b[..n].to_vec()], &vec![n, n]);
  test_basic_block_with_dims::<CQLinBasicBlock>(srs, &a, &vec![b[..m].to_vec()], &vec![m, N/m]);
}
