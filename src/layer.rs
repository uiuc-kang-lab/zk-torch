#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::basic_block::BasicBlock;
use crate::graph::Node;
pub use add::AddLayer;
pub use softmax::SoftmaxLayer;
use std::collections::HashMap;

use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub mod add;
pub mod softmax;

pub struct LayerConfig {
  pub input_params: HashMap<String, usize>,
}

pub trait Layer {
  fn load_onnx_layer(&self, config: &LayerConfig) -> Vec<Node> {
    vec![] // vec of nodes
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![]
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (0, 0)
  }

  fn run(
    &self,
    nodes: &Vec<Node>,
    inputs: &Vec<&ArrayD<Fr>>,
    models: &Vec<&ArrayD<Fr>>,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
  ) -> Vec<Vec<ArrayD<Fr>>> {
    let mut outputs = vec![vec![]; nodes.len()];
    nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
      outputs[i] = basic_blocks[n.basic_block].run(&models[n.basic_block], &myInputs);
    });
    return outputs;
  }

  fn prove(
    &self,
    nodes: &mut &Vec<Node>,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    basic_blocks: &mut Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs: Vec<&ArrayD<Data>> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        println!("proving {i} {:?}", n.basic_block);
        basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &myInputs, outputs[i], rng)
      })
      .collect()
  }

  fn verify(
    &self,
    nodes: &Vec<Node>,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) {
    nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng)
      })
      .collect()
  }
}

pub struct CustomLayer {
  pub nodes: Vec<usize>,
  pub inputs: Vec<Vec<(i32, usize)>>,
  pub output_node: (usize, usize),
}

impl Layer for CustomLayer {
  fn load_onnx_layer(&self, config: &LayerConfig) -> Vec<Node> {
    let layer = self.nodes.iter().zip(&self.inputs).map(|(x, y)| Node { basic_block: *x, inputs: y.to_vec() }).collect();
    layer
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    self.output_node
  }
}
