#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::{Graph, Node};
use ndarray::{ArrayD, IxDyn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use util::convert_to_data;
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
      Box::new(CQLinBasicBlock),
      Box::new(ReLUBasicBlock { input_SF: 1, output_SF: 1 }),
      Box::new(CQ2BasicBlock { table_dict: HashMap::new() }),
    ],
    nodes: vec![
      Node {
        basic_block: 0,
        inputs: vec![(-1, 0)],
      },
      Node {
        basic_block: 1,
        inputs: vec![(0, 0)],
      },
      Node {
        basic_block: 2,
        inputs: vec![(0, 0), (1, 0)],
      },
    ],
  };

  const m: usize = 1 << 4;
  const n: usize = 1 << 2;
  let matrix: Vec<_> = (0..n * m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-2..2))).collect();
  let matrix = ArrayD::from_shape_vec(vec![n, m], matrix).unwrap();
  let input: Vec<_> = (0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let input = ArrayD::from_shape_vec(vec![1, n], input).unwrap();

  //Run:
  let inputs = vec![&input];
  let empty = ArrayD::zeros(IxDyn(&[0]));
  let relu_cq_table = util::gen_cq_table(&graph.basic_blocks[1], -(1 << 5), 1 << 6);
  let models = vec![&matrix, &empty, &relu_cq_table];
  let outputs = graph.run(&inputs, &models);
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();

  //Setup:
  let models: Vec<ArrayD<Data>> = models.iter().map(|model| convert_to_data(srs, model)).collect();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = graph.setup(srs, &models);
  //Converting to affine
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();

  //Prove:
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<Data>>> = graph.encodeOutputs(srs, &models, &inputs, &outputs);
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  //Converting to affine
  let proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    proofs.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();

  //Verify:
  let models: Vec<ArrayD<DataEnc>> = models.iter().map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let models: Vec<&ArrayD<DataEnc>> = models.iter().map(|model| model).collect();
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  let outputs: Vec<Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
