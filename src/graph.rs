use crate::basic_block::BasicBlock;
use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G2Affine};
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub struct Node {
  pub basic_block: usize,
  pub input_nodes: Vec<usize>,
  pub output_nodes: Vec<usize>,
}

pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
  pub input_node: usize,
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let mut outputs = vec![ArrayD::zeros(vec![]); self.nodes.len()];
    // Run the nodes that have no inputs
    for i in 0..self.nodes.len() {
      if self.nodes[i].input_nodes.len() == 0 && i != self.input_node {
        outputs[i] = self.basic_blocks[self.nodes[i].basic_block].run(&models[self.nodes[i].basic_block], &vec![]);
      }
    }
    // DFS:
    let mut stack = vec![self.input_node];
    while stack.len() > 0 {
      let curr = stack.pop().unwrap();
      let currNode = &self.nodes[curr];
      if curr == self.input_node {
        outputs[curr] = self.basic_blocks[currNode.basic_block].run(&models[currNode.basic_block], inputs);
      } else {
        let myInputs = currNode.input_nodes.iter().map(|i| &(outputs[*i])).collect();
        outputs[curr] = self.basic_blocks[currNode.basic_block].run(&models[currNode.basic_block], &myInputs);
      }
      for n in &currNode.output_nodes {
        if *self.nodes[*n].input_nodes.last().unwrap() == curr {
          stack.push(*n);
        }
      }
    }
    return outputs;
  }
  pub fn setup(&self, srs: (&Vec<G1Affine>, &Vec<G2Affine>), models: &Vec<&Data>) -> Vec<(Vec<G1Affine>, Vec<G2Affine>)> {
    self.basic_blocks.iter().zip(models.iter()).map(|(b, m)| b.setup(srs, m)).collect()
  }
  pub fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&Data>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Data>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Affine>, Vec<G2Affine>)> {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        if i == self.input_node {
          self.basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &inputs, outputs[i], rng)
        } else {
          let inputs = n.input_nodes.iter().map(|j| outputs[*j]).collect();
          self.basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &inputs, outputs[i], rng)
        }
      })
      .collect()
  }
  pub fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    models: &Vec<&DataEnc>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&DataEnc>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    rng: &mut StdRng,
  ) {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        if i == self.input_node {
          self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], inputs, outputs[i], proofs[i], rng)
        } else {
          let inputs = n.input_nodes.iter().map(|j| outputs[*j]).collect();
          self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &inputs, outputs[i], proofs[i], rng)
        }
      })
      .collect()
  }
}
