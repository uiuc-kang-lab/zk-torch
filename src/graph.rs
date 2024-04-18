use crate::{basic_block::*, Layer, LayerConfig};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
}

pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub layers: Vec<Box<dyn Layer>>,
  pub layer_configs: Vec<LayerConfig>,
  pub layer_inputs: Vec<Vec<(i32, usize)>>, // ONNX graph (layer id, layer slot)
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Vec<Vec<Vec<ArrayD<Fr>>>> {
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
      
      let layer_nodes = s.load_onnx_layer(&self.layer_configs[i]);
      outputs[i] = s.run(&layer_nodes, &myInputs, models, &self.basic_blocks);
    });

    return outputs;
  }

  pub fn setup(&self, srs: &SRS, models: &Vec<&ArrayD<Data>>) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self.basic_blocks.iter().zip(models.iter()).map(|(b, m)| b.setup(srs, *m)).collect()
  }

  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&Vec<&ArrayD<Data>>>>,
    rng: &mut StdRng,
  ) -> Vec<Vec<(Vec<G1Projective>, Vec<G2Projective>)>> {
    self
      .layers
      .iter()
      .enumerate()
      .map(|(i, s)| {
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
        
        let layer_nodes = s.load_onnx_layer(&self.layer_configs[i]);
        s.prove(
          &mut &layer_nodes,
          srs,
          &setups,
          models,
          &myInputs,
          outputs[i],
          &mut self.basic_blocks,
          rng,
        )
      })
      .collect()
  }

  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
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
        
        let layer_nodes = s.load_onnx_layer(&self.layer_configs[i]);
        s.verify(
          &mut &layer_nodes,
          srs,
          models,
          &myInputs,
          outputs[i],
          proofs[i],
          &self.basic_blocks,
          rng,
        )
      })
      .collect()
  }
}
