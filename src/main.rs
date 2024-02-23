#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use basic_block::*;
use graph::{Graph, Node};
use ndarray::{arr1, ArrayD};
use rand::Rng;
use rayon::prelude::*;
mod basic_block;
mod graph;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn main() {
  let srs = ptau::load_file("challenge", 7);
  let srs = (&srs.0, &srs.1);
  let graph = Graph {
    basic_blocks: vec![
      Box::new(CQLinBasicBlock),
      Box::new(ReLUBasicBlock),
      Box::new(ConstBasicBlock),
      Box::new(MulBasicBlock),
      Box::new(AddBasicBlock),
      Box::new(CQBasicBlock),
    ],
    nodes: vec![
      Node {
        basic_block: 0,
        input_nodes: vec![],
        output_nodes: vec![1, 4],
      },
      Node {
        basic_block: 1,
        input_nodes: vec![0],
        output_nodes: vec![3],
      },
      Node {
        basic_block: 2,
        input_nodes: vec![],
        output_nodes: vec![3],
      },
      Node {
        basic_block: 3,
        input_nodes: vec![2, 1],
        output_nodes: vec![4],
      },
      Node {
        basic_block: 4,
        input_nodes: vec![0, 3],
        output_nodes: vec![5],
      },
      Node {
        basic_block: 5,
        input_nodes: vec![4],
        output_nodes: vec![],
      },
    ],
    input_node: 0,
  };

  const N: usize = 1 << 6;
  const m: usize = 1 << 4;
  let matrix: Vec<_> = (0..N).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-2..2))).collect();
  let input: Vec<_> = (0..m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();

  let matrix = ArrayD::from_shape_vec(vec![m, N / m], matrix).unwrap();
  let input = arr1(&input).into_dyn();
  let empty = ArrayD::zeros(vec![]);
  let constant = arr1(&vec![Fr::from(1 << 6); 1 << 2]).into_dyn();
  let relu_cq_table = util::gen_cq_table(&(*(graph.basic_blocks[1])), 1 << 6);
  let outputs = graph.run(&vec![&input], &vec![&matrix, &empty, &constant, &empty, &empty, &relu_cq_table]);

  let relu_cq_table = relu_cq_table.clone().into_raw_vec();
  for x in outputs[4].clone().into_raw_vec() {
    assert!(relu_cq_table.contains(&x), "{:?}", x);
  }

  //setup over basic_blocks
  //prove over nodes
  //verify over nodes
}
