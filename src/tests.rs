use crate::basic_block::*;
use crate::ptau;
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

fn testBasicBlock<BB: BasicBlock>(mut basic_block: BB, srs: &SRS, model: &Vec<&Vec<Fr>>, inputs: &Vec<&Vec<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let outputs = basic_block.run(model, inputs);
  let model: Vec<_> = model.iter().map(|x| Data::new(srs, x)).collect();
  let model = model.iter().map(|x| x).collect();
  let setup = basic_block.setup(srs, &model);
  let setup: (Vec<G1Affine>, Vec<G2Affine>) = (
    setup.0.iter().map(|y| (*y).into()).collect(),
    setup.1.iter().map(|y| (*y).into()).collect(),
  );
  let setup = (&setup.0, &setup.1);
  let inputs: Vec<_> = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| Data::new(srs, x)).collect();
  let outputs = outputs.iter().map(|x| x).collect();
  let mut rng2 = rng.clone();
  let proof = basic_block.prove(srs, setup, &model, &inputs, &outputs, &mut rng);
  let proof: (Vec<G1Affine>, Vec<G2Affine>) = (
    proof.0.iter().map(|y| (*y).into()).collect(),
    proof.1.iter().map(|y| (*y).into()).collect(),
  );
  let proof = (&proof.0, &proof.1);
  let model: Vec<_> = model.iter().map(|x| DataEnc::new(srs, x)).collect();
  let model = model.iter().map(|x| x).collect();
  let inputs: Vec<_> = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let outputs = outputs.iter().map(|x| x).collect();
  basic_block.verify(srs, &model, &inputs, &outputs, proof, &mut rng2);
}

#[test]
fn testBasicBlocks() {
  let srs = &ptau::load_file("challenge", 7);
  let N: usize = 1 << 6;
  let n: usize = 1 << 3;
  let a: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let b: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  testBasicBlock(EqBasicBlock {}, srs, &vec![], &vec![&a, &a]);
  testBasicBlock(AddBasicBlock {}, srs, &vec![], &vec![&a, &b]);
  testBasicBlock(AddBasicBlock {}, srs, &vec![], &vec![&a, &vec![b[0]]]);
  testBasicBlock(AddBasicBlock {}, srs, &vec![], &vec![&vec![a[0]], &b]);
  testBasicBlock(SubBasicBlock {}, srs, &vec![], &vec![&a, &b]);
  testBasicBlock(SubBasicBlock {}, srs, &vec![], &vec![&a, &vec![b[0]]]);
  testBasicBlock(SubBasicBlock {}, srs, &vec![], &vec![&vec![a[0]], &b]);
  testBasicBlock(MulBasicBlock {}, srs, &vec![], &vec![&a, &b]);
  testBasicBlock(MulScalarBasicBlock {}, srs, &vec![], &vec![&a, &vec![b[0]]]);
  testBasicBlock(MulConstBasicBlock { c: 12345 }, srs, &vec![], &vec![&a]);
  testBasicBlock(CQBasicBlock { table_dict: HashMap::new() }, srs, &vec![&a], &vec![&a[..n].to_vec()]);
  testBasicBlock(
    CQ2BasicBlock { table_dict: HashMap::new() },
    srs,
    &vec![&a, &b],
    &vec![&a[..n].to_vec(), &b[..n].to_vec()],
  );

  let l: usize = 1 << 4;
  let m: usize = 1 << 3;
  let n: usize = 1 << 2;
  let mut inputs: Vec<Vec<Fr>> = vec![];
  for _ in 0..l + n {
    inputs.push((0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect());
  }
  let inputs: Vec<_> = inputs.iter().map(|x| x).collect();
  testBasicBlock(MatMulBasicBlock { l: l }, srs, &vec![], &inputs);
  testBasicBlock(TransposeBasicBlock {}, srs, &vec![], &inputs[l..].to_vec());
  testBasicBlock(SumBasicBlock {}, srs, &vec![], &inputs[l..].to_vec());
  testBasicBlock(ConcatBasicBlock {}, srs, &vec![], &inputs[l..].to_vec());
  let intertwined = (CombineBasicBlock {}).run(&vec![], &vec![inputs[0], inputs[1]]);
  let intertwined = &intertwined[0];
  testBasicBlock(AlternatingBasicBlock {}, srs, &vec![], &vec![inputs[0], inputs[1], intertwined]);
  let split = (SplitBasicBlock {}).run(&vec![], &vec![inputs[0]]);
  let split = (&split[0], &split[1]);
  testBasicBlock(AlternatingBasicBlock {}, srs, &vec![], &vec![split.0, split.1, inputs[0]]);

  let mut matrix: Vec<Vec<Fr>> = vec![];
  for _ in 0..m {
    matrix.push((0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect());
  }
  let matrix: Vec<_> = matrix.iter().map(|x| x).collect();
  let mut inputs: Vec<Vec<Fr>> = vec![];
  for _ in 0..l {
    inputs.push((0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect());
  }
  let inputs: Vec<_> = inputs.iter().map(|x| x).collect();
  testBasicBlock(CQLinBasicBlock {}, srs, &matrix, &inputs);
}
