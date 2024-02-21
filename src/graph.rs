use crate::basic_block::BasicBlock;
use ark_bn254::Fr;
use ndarray::ArrayD;

struct Node {
  basic_block: usize,
  input_nodes: Vec<usize>,
  output_nodes: Vec<usize>,
}

pub struct Graph {
  basic_blocks: Vec<Box<dyn BasicBlock>>,
  nodes: Vec<Node>,
  input_node: usize,
}

impl Graph {
  pub fn run(&self, inputs: Vec<ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let outputs = vec![Vec::<ArrayD<Fr>>::new(); self.nodes.len()];
    return Vec::new();
  }
}
