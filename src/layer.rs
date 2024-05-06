#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::graph::{Node, Setup};
use crate::util::convert_to_data;
use crate::{basic_block::BasicBlock, graph::SetupType};
pub use add::AddLayer;
pub use cqlin::CQLinLayer;
pub use softmax::SoftmaxLayer;
use std::collections::HashMap;

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
}

pub trait Layer {
  fn layer_inputs(&self) -> Vec<Vec<(i32, usize)>> {
    vec![]
  }

  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
    let basic_block_map: HashMap<&Box<dyn BasicBlock>, usize> = basic_blocks.iter().enumerate().map(|(i, b)| (b, i)).collect();
    let used_blocks: Vec<Box<dyn BasicBlock>> = self.consume_basic_block(config);

    let nodes: Vec<usize> = used_blocks.iter().map(|b| *basic_block_map.get(b).unwrap()).collect();

    let inputs = self.layer_inputs();

    nodes.iter().zip(inputs).map(|(x, y)| Node { basic_block: *x, inputs: y }).collect()
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![]
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (0, 0)
  }

  fn run(
    &self,
    inputs: &Vec<&ArrayD<Fr>>,
    weights_map: &HashMap<String, ArrayD<Fr>>,
    layer_config: &LayerConfig,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
  ) -> Vec<Vec<ArrayD<Fr>>> {
    let nodes = self.load_layer_nodes(layer_config, basic_blocks);
    let mut outputs = vec![vec![]; nodes.len()];
    let empty = &ArrayD::zeros(IxDyn(&[0]));
    nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let bb = &basic_blocks[n.basic_block];
      let inputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
      let weight = if bb.weights_name().is_ok() {
        if let Some(s) = weights_map.get(&bb.weights_name().unwrap()) {
          s
        } else {
          panic!("Weight is missing from setups");
        }
      } else {
        empty
      };
      outputs[i] = basic_blocks[n.basic_block].run(&weight, &inputs);
    });

    outputs
  }

  fn encodeOutputs(
    &self,
    srs: &SRS,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<Vec<&ArrayD<Fr>>>,
    layer_config: &LayerConfig,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
  ) -> Vec<Vec<ArrayD<Data>>> {
    let nodes = self.load_layer_nodes(&layer_config, &basic_blocks);
    let mut outputsEnc = vec![vec![]; nodes.len()];
    nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let inputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputsEnc[*j as usize][*k]) }).collect();
      outputsEnc[i] = basic_blocks[n.basic_block].encodeOutputs(&srs, &inputs, &outputs[i]);
    });
    outputsEnc
  }

  fn prove(
    &self,
    srs: &SRS,
    setups: &Setup,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    layer_config: &LayerConfig,
    basic_blocks: &mut Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    let nodes = self.load_layer_nodes(layer_config, basic_blocks);
    nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let inputs: Vec<&ArrayD<Data>> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        let empty = convert_to_data(&srs, &ArrayD::zeros(IxDyn(&[0])));

        println!("proving {i} {:?}", n.basic_block);
        let bb = &mut basic_blocks[n.basic_block];
        let setup = if bb.weights_name().is_ok() {
          if let Some(s) = setups.weights.get(&bb.weights_name().unwrap()) {
            s
          } else {
            panic!("Weight is missing from setups");
          }
        } else if let Some(s) = setups.tables.get(&bb.name()) {
          s
        } else {
          &SetupType::None
        };
        bb.prove(srs, &setup, &inputs, outputs[i], rng)
      })
      .collect()
  }

  fn verify(
    &self,
    srs: &SRS,
    weights: &HashMap<String, ArrayD<DataEnc>>,
    tables: &HashMap<String, ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    layer_config: &LayerConfig,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) -> Vec<PairingCheck> {
    let nodes = self.load_layer_nodes(layer_config, basic_blocks);
    nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        let bb = &basic_blocks[n.basic_block];
        let empty = convert_to_data(&srs, &ArrayD::zeros(IxDyn(&[0]))).map(|x| DataEnc::new(srs, x));
        let model = if bb.weights_name().is_ok() {
          if let Some(s) = weights.get(&bb.weights_name().unwrap()) {
            s
          } else {
            panic!("Weight is missing from weights DataEnc map");
          }
        } else if let Some(s) = tables.get(&bb.name()) {
          s
        } else {
          &empty
        };
        println!("verifying {i} {:?}", n.basic_block);
        bb.verify(srs, model, &myInputs, outputs[i], proofs[i], rng)
      })
      .flatten()
      .collect()
  }
}
