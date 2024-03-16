use crate::basic_block::BasicBlock;
use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use rand::rngs::StdRng;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
  pub output_nodes: Vec<usize>,
}

pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
  pub input_nodes: Vec<usize>,
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&Vec<Fr>>, models: &Vec<&Vec<&Vec<Fr>>>) -> Vec<Vec<Vec<Fr>>> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    // Run the nodes that have no inputs
    for i in 0..self.nodes.len() {
      if self.nodes[i].inputs.len() == 0 {
        println!("running {i}");
        outputs[i] = self.basic_blocks[self.nodes[i].basic_block].run(&models[self.nodes[i].basic_block], &vec![]);
      }
    }
    // DFS:
    let mut stack = self.input_nodes.clone();
    while stack.len() > 0 {
      let curr = stack.pop().unwrap();
      let currNode = &self.nodes[curr];
      let myInputs = currNode.inputs.iter().map(|(i, j)| if *i < 0 { inputs[*j] } else { &(outputs[*i as usize][*j]) }).collect();
      println!("running {}",currNode.basic_block);
      outputs[curr] = self.basic_blocks[currNode.basic_block].run(&models[currNode.basic_block], &myInputs);
      for n in &currNode.output_nodes {
        if self.nodes[*n].inputs.last().unwrap().0 == (curr as i32) {
          stack.push(*n);
        }
      }
    }
    return outputs;
  }
  pub fn setup(&self, srs: &SRS, models: &Vec<&Vec<&Data>>) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self.basic_blocks.iter().zip(models.iter()).map(|(b, m)| b.setup(srs, m)).collect()
  }
  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&Vec<&Data>>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Vec<&Data>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs: Vec<_> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        println!("proving {i} {}",myInputs.len());
        self.basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &myInputs, outputs[i], rng)
      })
      .collect()
  }
  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&Vec<&DataEnc>>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&Vec<&DataEnc>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    rng: &mut StdRng,
  ) {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng)
      })
      .collect()
  }
}
