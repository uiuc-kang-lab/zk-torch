use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
}

pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub subgraphs: Vec<Subgraph>,
  pub subgraph_inputs: Vec<Vec<(i32, usize)>>, // ONNX graph (subgraph id, subgraph slot)
  pub output_in_subgraph: Vec<(usize, usize)>, // In subgraph, specify which node is the subgraph output (output node, output #)
}

pub struct Subgraph {
  pub nodes: Vec<Node>,
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Vec<Vec<Vec<ArrayD<Fr>>>> {
    let mut outputs: Vec<Vec<Vec<ArrayD<Fr>>>> = vec![vec![]; self.subgraphs.len()];

    self.subgraphs.iter().enumerate().for_each(|(i, s)| {
      let myInputs = self.subgraph_inputs[i]
        .iter()
        .map(|(j, k)| {
          if *j < 0 {
            inputs[*k]
          } else {
            &(outputs[*j as usize][self.output_in_subgraph[*j as usize].0][self.output_in_subgraph[*j as usize].1])
          }
        })
        .collect();

      outputs[i] = s.run(&myInputs, models, &self.basic_blocks);
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
      .subgraphs
      .iter_mut()
      .enumerate()
      .map(|(i, s)| {
        let myInputs = self.subgraph_inputs[i]
          .iter()
          .map(|(j, k)| {
            if *j < 0 {
              inputs[*k]
            } else {
              &(outputs[*j as usize][self.output_in_subgraph[*j as usize].0][self.output_in_subgraph[*j as usize].1])
            }
          })
          .collect();

        s.prove(srs, &setups, models, &myInputs, outputs[i], &mut self.basic_blocks, rng)
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
      .subgraphs
      .iter()
      .enumerate()
      .map(|(i, s)| {
        let myInputs = self.subgraph_inputs[i]
          .iter()
          .map(|(j, k)| {
            if *j < 0 {
              inputs[*k]
            } else {
              &(outputs[*j as usize][self.output_in_subgraph[*j as usize].0][self.output_in_subgraph[*j as usize].1])
            }
          })
          .collect();

        s.verify(srs, models, &myInputs, outputs[i], proofs[i], &self.basic_blocks, rng)
      })
      .collect()
  }
}

impl Subgraph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Vec<ArrayD<Fr>>> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
      outputs[i] = basic_blocks[n.basic_block].run(&models[n.basic_block], &myInputs);
    });
    return outputs;
  }

  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    basic_blocks: &mut Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs: Vec<&ArrayD<Data>> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        println!("proving {i} {:?}", n.basic_block);
        basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &myInputs, outputs[i], rng)
      })
      .collect()
  }

  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
    rng: &mut StdRng,
  ) {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng)
      })
      .collect()
  }
}
