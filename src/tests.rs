use crate::basic_block::*;
use crate::ptau;
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

fn testBasicBlock<BB: BasicBlock>(basic_block: BB, srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let output = basic_block.run(model, inputs);
  let model = Data::new(srs, model);
  let setup = basic_block.setup(srs, &model);
  let inputs: Vec<_> = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();
  let prove_start = Instant::now();
  let proof = basic_block.prove(srs, (&(setup.0), &(setup.1)), &model, &inputs, &output, &mut rng);
  let prove_duration = prove_start.elapsed();
  println!("Proving time: {:?}", prove_duration);
  let model = DataEnc::new(srs, &model);
  let inputs: Vec<_> = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let output = DataEnc::new(srs, &output);
  basic_block.verify(srs, &model, &inputs, &output, (&(proof.0), &(proof.1)), &mut rng2);
}

fn testMatMult(srs: (&Vec<G1Affine>, &Vec<G2Affine>), inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let matmultblock = MatMultBasicBlock {};
  let output = matmultblock.run(inputs);

  let inputs: Vec<_> = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let inputs: Vec<_> = inputs.iter().map(|x| x).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();

  // time it
  let start = Instant::now();
  let proof = matmultblock.prove(srs, inputs.clone(), &output, &mut rng);
  let duration = start.elapsed();
  println!("Proving time: {:?}", duration);

  matmultblock.verify(srs, inputs, &proof, &mut rng2);
}

#[test]
fn testBasicBlocks() {
  let srs = ptau::load_file("challenge_18", 18);
  let srs = (&srs.0, &srs.1);
  const N: usize = 1 << 6;
  const n: usize = 1 << 3;
  const m1: usize = 1 << 2;
  const m2: usize = 1 << 4;
  let a: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let a1 = a.clone();
  let b: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  testBasicBlock(
    AddBasicBlock {},
    srs,
    &arr1(&vec![]).into_dyn(),
    &vec![&arr1(&a).into_dyn(), &arr1(&b).into_dyn()],
  );
  testBasicBlock(
    MulBasicBlock {},
    srs,
    &arr1(&vec![]).into_dyn(),
    &vec![&arr1(&a).into_dyn(), &arr1(&b).into_dyn()],
  );
  testBasicBlock(CQBasicBlock {}, srs, &arr1(&a).into_dyn(), &vec![&arr1(&a[..n]).into_dyn()]);
  testBasicBlock(
    CQLinBasicBlock {},
    srs,
    &ArrayD::from_shape_vec(vec![m1, N / m1], a).unwrap(),
    &vec![&arr1(&b[..m1]).into_dyn()],
  );
  testBasicBlock(
    CQLinBasicBlock {},
    srs,
    &ArrayD::from_shape_vec(vec![m2, N / m2], a1).unwrap(),
    &vec![&arr1(&b[..m2]).into_dyn()],
  );

  let mat_a_tensor = ArrayD::from_shape_vec(
    vec![128, 128],
    (0..128 * 128).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect(),
  )
  .unwrap();
  let mat_b_tensor = ArrayD::from_shape_vec(
    vec![128, 128],
    (0..128 * 128).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect(),
  )
  .unwrap();
  testMatMult(srs, &vec![&mat_a_tensor, &mat_b_tensor]);
  
  let a: Vec<_> = (0..128 * 128).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let b: Vec<_> = (0..128 * 128).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  testBasicBlock(
    BridgeBasicBlock {},
    srs,
    &arr1(&a).into_dyn(),
    &vec![&arr1(&a).into_dyn(), &arr1(&b).into_dyn()],
  );
}
