#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use basic_block::*;
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
mod basic_block;
mod ptau;
mod util;

fn test_basic_block<BB: BasicBlock>(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &ArrayD<Fr>, inputs: &Vec<ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let output = BB::run(model, inputs);
  let model = Data::new(srs, model);
  let setup = BB::setup(srs, &model);
  let inputs = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs, (&(setup.0), &(setup.1)), &model, &inputs, &output, &mut rng);
  let model = DataEnc::new(srs, &model);
  let inputs = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let output = DataEnc::new(srs, &output);
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
  test_basic_block::<AddBasicBlock>(srs, &arr1(&vec![]).into_dyn(), &vec![arr1(&a).into_dyn(), arr1(&b).into_dyn()]);
  test_basic_block::<MulBasicBlock>(srs, &arr1(&vec![]).into_dyn(), &vec![arr1(&a).into_dyn(), arr1(&b).into_dyn()]);
  test_basic_block::<CQBasicBlock>(srs, &arr1(&a).into_dyn(), &vec![arr1(&a[..n]).into_dyn()]);
  test_basic_block::<CQLinBasicBlock>(srs, &ArrayD::from_shape_vec(vec![m, N / m], a).unwrap(), &vec![arr1(&b[..m]).into_dyn()]);
}
