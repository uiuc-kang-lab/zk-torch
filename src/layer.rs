#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::graph::Node;
use crate::util::convert_to_data;
use crate::{basic_block::BasicBlock, setup::Setup};
pub use add::AddLayer;
pub use cqlin::CQLinLayer;
pub use softmax::SoftmaxLayer;
use std::collections::HashMap;
use std::rc::Rc;

use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{ArrayD, IxDyn};
use rand::rngs::StdRng;

pub mod add;
pub mod cqlin;
pub mod softmax;

#[derive(Debug)]
pub enum LayerType {
  Add,
  CQLin,
  Softmax,
}

#[derive(Debug)]
pub struct LayerConfig {
  pub layer_type: LayerType,
  pub input_params: HashMap<String, usize>,
  pub weights_names: Vec<String>,
  // pub table_names: Vec<String>,
}

pub trait Layer {
  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
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
    weights: &Vec<Rc<ArrayD<Fr>>>,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
  ) -> Vec<Vec<ArrayD<Fr>>> {
    let mut outputs = vec![vec![]; nodes.len()];
    nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let inputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
      outputs[i] = basic_blocks[n.basic_block].run(&weights[n.basic_block], &inputs);
    });
    return outputs;
  }

  fn prove(
    &self,
    nodes: &mut &Vec<Node>,
    srs: &SRS,
    setups: &Setup,
    // models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    basic_blocks: &mut Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let inputs: Vec<&ArrayD<Data>> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        let empty = convert_to_data(&srs, &ArrayD::zeros(IxDyn(&[0])));

        println!("proving {i} {:?}", n.basic_block);
        let bb = &mut basic_blocks[n.basic_block];
        let setup = (setups.weights.get(&bb.weights_name()), setups.tables.get(&bb.name()));
        bb.prove(srs, &setup, &inputs, outputs[i], rng)
      })
      .collect()
  }

  fn verify(
    &self,
    nodes: &Vec<Node>,
    srs: &SRS,
    weights: &HashMap<String, ArrayD<DataEnc>>,
    tables: &HashMap<String, ArrayD<DataEnc>>,
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
        let bb = &basic_blocks[n.basic_block];
        let setup = (weights.get(&bb.weights_name()), tables.get(&bb.name()));
        let empty = convert_to_data(&srs, &ArrayD::zeros(IxDyn(&[0]))).map(|x| DataEnc::new(srs, x));
        let model = if let Some(s) = setup.0 {
          s
        } else if let Some(s) = setup.1 {
          s
        } else {
          &empty
        };
        bb.verify(srs, model, &myInputs, outputs[i], proofs[i], rng)
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
  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
    self
      .nodes
      .iter()
      .zip(&self.inputs)
      .map(|(x, y)| Node {
        basic_block: *x,
        inputs: y.to_vec(),
      })
      .collect()
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    self.output_node
  }
}
