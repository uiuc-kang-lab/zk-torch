use std::{collections::HashMap, rc::Rc};

use crate::{basic_block::*, setup::Setup, util, AddLayer, CQLinLayer, Layer, LayerConfig, LayerType, SoftmaxLayer};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{ArrayD, IxDyn};
use rand::rngs::StdRng;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
}

pub struct Graph {
  pub layers: Vec<Box<dyn Layer>>,
  pub layer_configs: Vec<LayerConfig>,
  pub layer_inputs: Vec<Vec<(i32, usize)>>, // ONNX graph (layer id, layer slot)
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub basic_block_weights: Vec<Rc<ArrayD<Fr>>>,
  pub weights_map: HashMap<String, Rc<ArrayD<Fr>>>,
  pub table_map: HashMap<String, Rc<ArrayD<Fr>>>,
}

impl Graph {
  fn layer_config_to_layer(config: &LayerConfig) -> Box<dyn Layer> {
    match config.layer_type {
      LayerType::Add => Box::new(AddLayer {}) as Box<dyn Layer>,
      LayerType::CQLin => Box::new(CQLinLayer {}) as Box<dyn Layer>,
      LayerType::Softmax => Box::new(SoftmaxLayer {}) as Box<dyn Layer>,
    }
  }

  pub fn new(
    configs: Vec<LayerConfig>,
    weights: &HashMap<String, Rc<ArrayD<Fr>>>,
    layer_inputs: Vec<Vec<(i32, usize)>>,
    table_offset: i32,
    table_size: usize,
  ) -> Self {
    let layers: Vec<_> = configs.iter().map(|c| Self::layer_config_to_layer(c)).collect();

    let mut basic_blocks = vec![];
    for (i, layer) in layers.iter().enumerate() {
      basic_blocks.append(&mut layer.consume_basic_block(&configs[i]));
    }

    let empty = Rc::new(ArrayD::zeros(IxDyn(&[0])));
    let mut table_map: HashMap<String, Rc<ArrayD<Fr>>> = HashMap::new();
    for i in 1..basic_blocks.len() {
      let table = match (basic_blocks[i - 1].block_type(), basic_blocks[i].block_type()) {
        (BasicBlockType::ChangeSF, BasicBlockType::CQ2)
        | (BasicBlockType::Exp, BasicBlockType::CQ2)
        | (BasicBlockType::Log, BasicBlockType::CQ2)
        | (BasicBlockType::ReLU, BasicBlockType::CQ2)
        | (BasicBlockType::Sqrt, BasicBlockType::CQ2) => {
          if !table_map.contains_key(&basic_blocks[i].name()) {
            Some(util::gen_cq_table(&basic_blocks[i - 1], table_offset, table_size))
          } else {
            None
          }
        }
        _ => None,
      };
      if let Some(t) = table {
        let temp = Rc::new(t);
        table_map.insert(basic_blocks[i].name(), temp);
      }
    }

    let basic_block_weights: Vec<_> = basic_blocks
      .iter()
      .map(|bb| {
        let name = bb.weights_name();
        if name.is_empty() {
          empty.clone()
        } else {
          weights.get(&name).unwrap().clone()
        }
      })
      .collect();

    Graph {
      layers,
      layer_configs: configs,
      layer_inputs,
      basic_blocks,
      basic_block_weights,
      weights_map: weights.clone(),
      table_map,
    }
  }

  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>) -> Vec<Vec<Vec<ArrayD<Fr>>>> {
    let mut outputs: Vec<Vec<Vec<ArrayD<Fr>>>> = vec![vec![]; self.layers.len()];

    self.layers.iter().enumerate().for_each(|(i, s)| {
      let myInputs = self.layer_inputs[i]
        .iter()
        .map(|(j, k)| {
          if *j < 0 {
            inputs[*k]
          } else {
            let output_in_layer = self.layers[*j as usize].layer_output_node(&self.layer_configs[*j as usize]);
            &(outputs[*j as usize][output_in_layer.0][output_in_layer.1])
          }
        })
        .collect();

      let layer_nodes = s.load_layer_nodes(&self.layer_configs[i], &self.basic_blocks);
      outputs[i] = s.run(&layer_nodes, &myInputs, &self.basic_block_weights, &self.basic_blocks);
    });

    return outputs;
  }

  pub fn setup(&self, srs: &SRS) -> Setup {
    Setup::new(&srs, &self.weights_map, &self.table_map)
  }

  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Setup,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&Vec<&ArrayD<Data>>>>,
    rng: &mut StdRng,
  ) -> Vec<Vec<(Vec<G1Projective>, Vec<G2Projective>)>> {
    self
      .layers
      .iter()
      .enumerate()
      .map(|(i, s)| {
        let inputs = self.layer_inputs[i]
          .iter()
          .map(|(j, k)| {
            if *j < 0 {
              inputs[*k]
            } else {
              let output_in_layer = self.layers[*j as usize].layer_output_node(&self.layer_configs[*j as usize]);
              &(outputs[*j as usize][output_in_layer.0][output_in_layer.1])
            }
          })
          .collect();

        let layer_nodes = s.load_layer_nodes(&self.layer_configs[i], &self.basic_blocks);
        s.prove(&mut &layer_nodes, srs, &setups, &inputs, outputs[i], &mut self.basic_blocks, rng)
      })
      .collect()
  }

  pub fn verify(
    &self,
    srs: &SRS,
    setups: &Setup,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&Vec<&ArrayD<DataEnc>>>>,
    proofs: &Vec<&Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>>,
    rng: &mut StdRng,
  ) {
    self
      .layers
      .iter()
      .enumerate()
      .map(|(i, s)| {
        let inputs = self.layer_inputs[i]
          .iter()
          .map(|(j, k)| {
            if *j < 0 {
              inputs[*k]
            } else {
              let output_in_layer = self.layers[*j as usize].layer_output_node(&self.layer_configs[*j as usize]);
              &(outputs[*j as usize][output_in_layer.0][output_in_layer.1])
            }
          })
          .collect();

        let layer_nodes = s.load_layer_nodes(&self.layer_configs[i], &self.basic_blocks);
        let weights_dataenc: HashMap<_, _> = setups.weights.iter().map(|(k, v)| (k.clone(), v.weights.map(|x| DataEnc::new(srs, x)))).collect();
        let table_dataenc: HashMap<_, _> = setups.tables.iter().map(|(k, v)| (k.clone(), v.table.map(|x| DataEnc::new(srs, x)))).collect();
        s.verify(
          &mut &layer_nodes,
          srs,
          &weights_dataenc,
          &table_dataenc,
          &inputs,
          outputs[i],
          proofs[i],
          &self.basic_blocks,
          rng,
        )
      })
      .collect()
  }
}
