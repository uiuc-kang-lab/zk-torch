use std::collections::{HashMap, HashSet};

use crate::{basic_block::*, util, AddLayer, CQLinLayer, Layer, LayerConfig, LayerType, SoftmaxLayer};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{ArrayD, IxDyn};
use rand::rngs::StdRng;

use self::ops::is_nonlinearity;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
}

pub struct CQLinSetup {
  pub weights: ArrayD<Data>,
  pub R: Vec<G1Affine>,
  pub Q: Vec<G1Affine>,
  pub S: Vec<G1Affine>,
  pub P_R: Vec<G1Affine>,
  pub L_V_i_x_n: Vec<G1Affine>,
  pub L_V_i_x: Vec<G1Affine>,
  pub L_H_i_x: Vec<G1Affine>,
  pub M_x: G2Affine,
}

pub struct CQSetup {
  pub table: ArrayD<Data>,
  pub Q_i_x_1: Vec<Vec<G1Affine>>,
  pub L_i_x_1: Vec<G1Affine>,
  pub L_i_0_x_1: Vec<G1Affine>,
  pub T_x_2: Vec<G2Affine>,
}

pub enum SetupType {
  CQLin(CQLinSetup),
  CQ(CQSetup),
  None,
}

pub struct Setup {
  pub weights: HashMap<String, SetupType>,
  pub tables: HashMap<String, SetupType>,
}
pub struct Graph {
  pub layers: Vec<Box<dyn Layer>>,
  pub layer_configs: Vec<LayerConfig>,
  pub layer_inputs: Vec<Vec<(i32, usize)>>, // ONNX graph (layer id, layer slot)
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub weights_map: HashMap<String, ArrayD<Fr>>,
  pub table_map: HashMap<String, ArrayD<Fr>>,
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
    weights: HashMap<String, ArrayD<Fr>>,
    layer_inputs: Vec<Vec<(i32, usize)>>,
    table_offset: i32,
    table_size: usize,
  ) -> Self {
    let layers: Vec<_> = configs.iter().map(|c| Self::layer_config_to_layer(c)).collect();

    let mut basic_blocks = vec![];
    for (i, layer) in layers.iter().enumerate() {
      basic_blocks.append(&mut layer.consume_basic_block(&configs[i]));
    }
    // Remove duplicates but keep order
    let mut seen = HashSet::new();
    basic_blocks.retain(|x| seen.insert(x.name()));

    let mut table_map: HashMap<String, ArrayD<Fr>> = HashMap::new();
    for i in 1..basic_blocks.len() {
      if is_nonlinearity(basic_blocks[i - 1].block_type()) {
        match basic_blocks[i].block_type() {
          BasicBlockType::CQ2 => {
            if !table_map.contains_key(&basic_blocks[i].name()) {
              table_map.insert(basic_blocks[i].name(), util::gen_cq_table(&basic_blocks[i - 1], table_offset, table_size));
            }
          }
          _ => panic!("Nonlinearity isn't followed by a lookup."),
        };
      }
    }

    Graph {
      layers,
      layer_configs: configs,
      layer_inputs,
      basic_blocks,
      weights_map: weights,
      table_map,
    }
  }

  // Returns outputs indexed by [layer no.][node id in that layer][output index of that node].
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
            println!("output_in_layer: {:?}", output_in_layer);
            &(outputs[*j as usize][output_in_layer.0][output_in_layer.1])
          }
        })
        .collect();

      let empty = &ArrayD::zeros(IxDyn(&[0]));
      let basic_block_weights: Vec<_> = self
        .basic_blocks
        .iter()
        .map(|bb| {
          if bb.weights_name().is_ok() {
            if let Some(s) = self.weights_map.get(&bb.weights_name().unwrap()) {
              s
            } else {
              panic!("Weight is missing from setups");
            }
          } else {
            empty
          }
        })
        .collect();

      let layer_nodes = s.load_layer_nodes(&self.layer_configs[i], &self.basic_blocks);
      outputs[i] = s.run(&layer_nodes, &myInputs, &basic_block_weights, &self.basic_blocks);
    });

    return outputs;
  }

  pub fn setup(&self, srs: &SRS) -> Setup {
    let cqlin_bb = Box::new(CQLinBasicBlock {
      weights_name: "".to_string(),
    });
    let weight_setups: HashMap<_, _> = self.weights_map.iter().map(|(k, v)| (k.clone(), cqlin_bb.setup(&srs, v))).collect();

    let cq2_bb = Box::new(CQ2BasicBlock {
      table_dict: HashMap::new(),
      name: "".to_string(),
    });
    let table_setups: HashMap<_, _> = self.table_map.iter().map(|(k, v)| (k.clone(), cq2_bb.setup(&srs, v))).collect();

    Setup {
      weights: weight_setups,
      tables: table_setups,
    }
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
        let weights_dataenc: HashMap<_, _> = setups
          .weights
          .iter()
          .map(|(k, v)| {
            if let SetupType::CQLin(setup) = v {
              (k.clone(), setup.weights.map(|x| DataEnc::new(srs, x)))
            } else {
              panic!("setups.weights has an incorrect SetupType")
            }
          })
          .collect();
        let table_dataenc: HashMap<_, _> = setups
          .tables
          .iter()
          .map(|(k, v)| {
            if let SetupType::CQ(setup) = v {
              (k.clone(), setup.table.map(|x| DataEnc::new(srs, x)))
            } else {
              panic!("setups.tables has an incorrect SetupType")
            }
          })
          .collect();
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
