use crate::{basic_block::*, Layer};
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
  pub layer_nodes: Vec<Vec<Node>>,          // each Vec<Node> is a layer
  pub layer_inputs: Vec<Vec<(i32, usize)>>, // ONNX graph (layer id, layer slot)
  pub output_in_layer: Vec<(usize, usize)>, // In layer, specify which node is the layer output (output node, output #)
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
            &(outputs[*j as usize][self.output_in_layer[*j as usize].0][self.output_in_layer[*j as usize].1])
          }
        })
        .collect();

      outputs[i] = s.run(&self.layer_nodes[i], &myInputs, models, &self.basic_blocks);
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
      .iter_mut()
      .enumerate()
      .map(|(i, s)| {
        let myInputs = self.layer_inputs[i]
          .iter()
          .map(|(j, k)| {
            if *j < 0 {
              inputs[*k]
            } else {
              &(outputs[*j as usize][self.output_in_layer[*j as usize].0][self.output_in_layer[*j as usize].1])
            }
          })
          .collect();

        s.prove(
          &mut &self.layer_nodes[i],
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
              &(outputs[*j as usize][self.output_in_layer[*j as usize].0][self.output_in_layer[*j as usize].1])
            }
          })
          .collect();

        s.verify(
          &mut &self.layer_nodes[i],
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
