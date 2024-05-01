#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::Graph;
use layer::*;
use ndarray::ArrayD;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use util::convert_to_data;

mod basic_block;
mod graph;
mod layer;

mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn main() {
  let srs = &ptau::load_file("challenge", 7);

  // create weights map
  let matrix: Vec<_> = (0..n * m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-2..2))).collect();
  let matrix = ArrayD::from_shape_vec(vec![n, m], matrix).unwrap();

  let weights = HashMap::from([("w1".to_string(), matrix)]);

  // create layer 0: a cqlin layer
  let cqlin_config = LayerConfig {
    layer_type: LayerType::CQLin,
    input_params: HashMap::from([("input_SF".to_string(), 1), ("output_SF".to_string(), 1)]),
    weights_names: vec!["w1".to_string()],
  };

  // create layer 1: a Softmax layer
  let softmax_config = LayerConfig {
    layer_type: LayerType::Softmax,
    input_params: HashMap::from([("input_SF".to_string(), 1), ("output_SF".to_string(), 1)]),
    weights_names: vec![],
  };

  // create layer 2: an Add layer
  let add_config = LayerConfig {
    layer_type: LayerType::Add,
    input_params: HashMap::new(),
    weights_names: vec![],
  };

  let mut graph = Graph::new(
    vec![cqlin_config, softmax_config, add_config],
    weights,
    vec![vec![(-1, 0)], vec![(0, 0)], vec![(-1, 1), (1, 0)]],
    -(1 << 5),
    1 << 6,
  );

  const m: usize = 1 << 4;
  const n: usize = 1 << 2;
  let input: Vec<_> = (0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let input = ArrayD::from_shape_vec(vec![1, n], input).unwrap();

  let adder_input = ArrayD::from_shape_vec(vec![m], vec![Fr::from(1); m]).unwrap();

  //Run:
  let inputs = vec![&input, &adder_input];

  let outputs = graph.run(&inputs);
  let outputs: Vec<Vec<Vec<&ArrayD<Fr>>>> = outputs.iter().map(|output| output.iter().map(|o| o.iter().map(|x| x).collect()).collect()).collect();

  //Setup:
  let setup = graph.setup(srs);

  //Prove:
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<Vec<ArrayD<Data>>>> = outputs
    .iter()
    .map(|outputs| outputs.iter().map(|output| output.iter().map(|o| convert_to_data(srs, o)).collect()).collect())
    .collect();
  let outputs: Vec<Vec<Vec<&ArrayD<Data>>>> =
    outputs.iter().map(|outputs| outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect()).collect();
  let outputs: Vec<Vec<&Vec<&ArrayD<Data>>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&Vec<&ArrayD<Data>>>> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setup, &inputs, &outputs, &mut rng);

  //Converting to affine
  let proofs: Vec<Vec<(Vec<G1Affine>, Vec<G2Affine>)>> = proofs
    .iter()
    .map(|proof| proof.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect())
    .collect();
  let proofs: Vec<Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>> = proofs.iter().map(|proof| proof.iter().map(|x| (&x.0, &x.1)).collect()).collect();
  let proofs = proofs.iter().map(|x| x).collect();

  //Verify:
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<Vec<ArrayD<DataEnc>>>> = outputs
    .iter()
    .map(|output| (**output).iter().map(|o| o.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect())
    .collect();
  let outputs: Vec<Vec<Vec<&ArrayD<DataEnc>>>> =
    outputs.iter().map(|output| output.iter().map(|o| o.iter().map(|x| x).collect()).collect()).collect();
  let outputs: Vec<Vec<&Vec<&ArrayD<DataEnc>>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&Vec<&ArrayD<DataEnc>>>> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &setup, &inputs, &outputs, &proofs, &mut rng2);
}
