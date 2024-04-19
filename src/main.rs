#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::Graph;
use layer::*;
use ndarray::{ArrayD, IxDyn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use util::convert_to_data;

mod basic_block;
mod graph;
mod layer;

//mod onnx_converter;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn main() {
  let srs = &ptau::load_file("challenge", 7);

  // create layer 0: a custom layer
  let mut basic_blocks: Vec<Box<dyn BasicBlock>> = vec![
    Box::new(CQLinBasicBlock),
    Box::new(ReLUBasicBlock { input_SF: 1, output_SF: 1 }),
    Box::new(CQ2BasicBlock {
      table_dict: HashMap::new(),
      name: "ReLU".to_string(),
    }),
    Box::new(SqueezeBasicBlock),
  ];

  let custom_layer = CustomLayer {
    nodes: vec![0, 1, 2, 3],
    inputs: vec![vec![(-1, 0)], vec![(0, 0)], vec![(0, 0), (1, 0)], vec![(1, 0)]],
    output_node: (3, 0),
  };

  // create layer 1: a Softmax layer
  let softmax_config = LayerConfig {
    input_params: HashMap::from([("input_SF".to_string(), 1), ("output_SF".to_string(), 1)]),
  };

  let softmax = SoftmaxLayer {};

  basic_blocks.append(&mut softmax.consume_basic_block(&softmax_config)); // we will need to handle repeated basic blocks later

  // create graph by combining layers
  let mut graph = Graph {
    basic_blocks: basic_blocks,
    layers: vec![Box::new(custom_layer), Box::new(softmax)],
    layer_configs: vec![
      LayerConfig {
        input_params: HashMap::new(),
      },
      softmax_config,
    ],
    layer_inputs: vec![vec![(-1, 0)], vec![(0, 0)]], // ONNX graph (layer id, layer slot), which can be loaded from ONNX file
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
  let exp_cq_table = util::gen_cq_table(&graph.basic_blocks[6], -(1 << 5), 1 << 6);
  let log_cq_table = util::gen_cq_table(&graph.basic_blocks[10], -(1 << 5), 1 << 6);
  let models = vec![
    &matrix,
    &empty,
    &relu_cq_table,
    &empty,
    &empty,
    &empty,
    &empty,
    &exp_cq_table,
    &empty,
    &empty,
    &empty,
    &log_cq_table,
    &empty,
  ];

  let outputs = graph.run(&inputs, &models);
  let outputs: Vec<Vec<Vec<&ArrayD<Fr>>>> = outputs.iter().map(|output| output.iter().map(|o| o.iter().map(|x| x).collect()).collect()).collect();

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
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);

  //Converting to affine
  let proofs: Vec<Vec<(Vec<G1Affine>, Vec<G2Affine>)>> = proofs
    .iter()
    .map(|proof| proof.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect())
    .collect();
  let proofs: Vec<Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>> = proofs.iter().map(|proof| proof.iter().map(|x| (&x.0, &x.1)).collect()).collect();
  let proofs = proofs.iter().map(|x| x).collect();

  //Verify:
  let models: Vec<ArrayD<DataEnc>> = models.iter().map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let models: Vec<&ArrayD<DataEnc>> = models.iter().map(|model| model).collect();
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
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
