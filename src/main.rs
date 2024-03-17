#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::{Graph, Node};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
mod basic_block;
mod graph;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn main() {
  let srs = &ptau::load_file("challenge", 7);
  let mut graph = Graph {
    basic_blocks: vec![
      Box::new(MatMulBasicBlock { l: 2 }),
      Box::new(ReLUBasicBlock),
      Box::new(CQ2BasicBlock { table_dict: HashMap::new() }),
    ],
    nodes: vec![
      Node {
        basic_block: 0,
        inputs: vec![(-1,0),(-1,1),(-1,2),(-1,3),(-1,4),(-1,5)],
        output_nodes: vec![1, 2],
      },
      Node {
        basic_block: 1,
        inputs: vec![(0, 1)],
        output_nodes: vec![2],
      },
      Node {
        basic_block: 2,
        inputs: vec![(0, 1), (1, 0)],
        output_nodes: vec![],
      },
    ],
    input_nodes: vec![0],
  };

  const m: usize = 1 << 4;
  const n: usize = 1 << 2;
  let matrix: Vec<Vec<_>> =
    (0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| (0..m).map(|_| Fr::from(rng.gen_range(-2..2))).collect()).collect();
  let mut matrix: Vec<_> = matrix.iter().map(|x| x).collect();
  let input: Vec<_> = (0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let input2: Vec<_> = (0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();

  //Run:
  let mut inputs = vec![&input, &input2];
  inputs.append(&mut matrix);
  let empty = vec![];
  let (id, relu_cq_table) = util::gen_cq_table(&graph.basic_blocks[1], 1 << 6, -( 1<< 5));
  let relu_cq_table = vec![&id, &relu_cq_table];
  let models = vec![&empty, &empty, &relu_cq_table];
  let outputs = graph.run(&inputs, &models);

  //Setup:
  let models: Vec<Vec<_>> = models.iter().map(|model| model.iter().map(|x| Data::new(srs, x)).collect()).collect();
  let models: Vec<_> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models = models.iter().map(|x| x).collect();
  let setups = graph.setup(srs, &models);
  //Converting to affine
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();

  //Prove:
  let inputs: Vec<_> = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| Data::new(srs, x)).collect()).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| x).collect()).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  //Converting to affine
  let proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    proofs.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();

  //Verify:
  let models: Vec<Vec<_>> = models.iter().map(|model| (**model).iter().map(|x| DataEnc::new(srs, *x)).collect()).collect();
  let models: Vec<_> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models = models.iter().map(|x| x).collect();
  let inputs: Vec<_> = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| DataEnc::new(srs, x)).collect()).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| x).collect()).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
