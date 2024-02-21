#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use basic_block::*;
use graph::{Graph, Node};
use ndarray::{arr1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
mod basic_block;
mod graph;
mod ptau;
mod util;

fn test_basic_block<BB: BasicBlock>(basic_block: BB, srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &ArrayD<Fr>, inputs: &Vec<ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let output = basic_block.run(model, inputs);
  let model = Data::new(srs, model);
  let setup = basic_block.setup(srs, &model);
  let inputs = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let output = Data::new(srs, &output);
  let mut rng2 = rng.clone();
  let proof = basic_block.prove(srs, (&(setup.0), &(setup.1)), &model, &inputs, &output, &mut rng);
  let model = DataEnc::new(srs, &model);
  let inputs = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let output = DataEnc::new(srs, &output);
  basic_block.verify(srs, &model, &inputs, &output, (&(proof.0), &(proof.1)), &mut rng2);
}
fn main() {
  let srs = ptau::load_file("challenge", 7);
  let srs = (&srs.0, &srs.1);
  let a: Vec<_> = (0..1 << 6).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::rand(rng)).collect();
  let g = Graph {
    basic_blocks: vec![Box::new(CQLinBasicBlock), Box::new(ReLUBasicBlock), Box::new(CQBasicBlock)],
    nodes: vec![
      Node {
        basic_block: 0,
        input_nodes: vec![],
        output_nodes: vec![1, 2],
      },
      Node {
        basic_block: 1,
        input_nodes: vec![0],
        output_nodes: vec![2],
      },
      Node {
        basic_block: 2,
        input_nodes: vec![0, 1],
        output_nodes: vec![],
      },
    ],
    input_node: 0,
  };
}
