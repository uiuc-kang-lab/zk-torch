use crate::basic_block::BasicBlock;
use ark_bn254::Fr;
use ndarray::ArrayD;

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
  pub fn run(&self, inputs: Vec<ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let outputs = vec![Vec::<ArrayD<Fr>>::new(); self.nodes.len()];
    return Vec::new();
  }
}
