#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use rand::{rngs::StdRng,SeedableRng};
use rayon::prelude::*;
use basic_block::*;
use ndarray::{Array, IxDyn};
mod basic_block;
mod util;
mod ptau;

fn test_basic_block<BB: BasicBlock>(srs: (&Vec<G1Affine>,&Vec<G2Affine>), model: &Vec<Tensor<Fr>>, inputs: &Vec<Tensor<Fr>>){
  let mut rng = StdRng::from_entropy();
  // Witness generation
  let output = BB::run(model,inputs);
  // One-time setup for model
  let model = Data::new(srs, &model[0]);
  let setup = BB::setup(srs,&model);
  // Prover time
  let inputs = inputs.iter().map(|x| Data::new(srs,x)).collect();
  let mut output_data = Data::new(srs,&Tensor::zeros(IxDyn(&[0])));
  if output.len() != 0 {
    output_data = Data::new(srs,&output[0]);
  }
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs,&setup,&model,&inputs,&output_data,&mut rng);
  let model = DataEnc::new(&model);
  let inputs = inputs.iter().map(|x| DataEnc::new(x)).collect();
  let output = DataEnc::new(&output_data);
  // Verifier time
  BB::verify(srs,&model,&inputs,&output,&proof,&mut rng2);
}
fn main() {
  let srs = ptau::load_file("challenge",7);
  let srs = (&srs.0,&srs.1);
  const N:usize = 1<<6;
  const n:usize = 1<<3;
  let a: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let b: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let a_tensor = Array::from_shape_vec(IxDyn(&[N]), a.clone()).unwrap();
  let b_tensor = Array::from_shape_vec(IxDyn(&[N]), b.clone()).unwrap();
  let c_tensor = Array::from_shape_vec(IxDyn(&[n, n]), b.clone()).unwrap();
  test_basic_block::<AddBasicBlock>(srs,&vec![a_tensor.clone()],&vec![a_tensor.clone(),b_tensor.clone()]);
  test_basic_block::<MulBasicBlock>(srs,&vec![a_tensor.clone()],&vec![a_tensor.clone(),b_tensor.clone()]);
  test_basic_block::<CQBasicBlock>(srs,&vec![a_tensor.clone()], &vec![Array::from_shape_vec(IxDyn(&[n]), a[..n].to_vec()).unwrap()]);
  test_basic_block::<CQLinBasicBlock>(srs,&vec![a_tensor.clone()],&vec![Array::from_shape_vec(IxDyn(&[n]), b[..n].to_vec()).unwrap()]);
  test_basic_block::<RopeBasicBlock>(srs,&vec![a_tensor.clone()],&vec![c_tensor.clone()]);
  test_basic_block::<TransposeBasicBlock>(srs,&vec![a_tensor.clone()],&vec![c_tensor.clone()]);
  test_basic_block::<MatMultBasicBlock>(srs,&vec![a_tensor.clone()],&vec![c_tensor.clone(),c_tensor.clone()]);
}
