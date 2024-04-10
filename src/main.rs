#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use batched_basic_block::*;
use graph::{Graph, Node};
use ndarray::{arr0, arr1, ArrayD, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
mod basic_block;
mod batched_basic_block;
mod graph;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn convert_to_data(srs: &SRS, a: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
  a.iter()
    .map(|x| {
      if x.ndim() == 1 {
        arr0(Data::new(srs, &(*x).clone().into_owned().into_raw_vec())).into_dyn()
      } else {
        x.map_axis(Axis(x.ndim() - 2), |y| Data::new(srs, &y.into_owned().into_raw_vec()))
      }
    })
    .collect()
}

fn main() {
  let srs = &ptau::load_file("challenge", 7);
  let mut graph = Graph {
    basic_blocks: vec![
      BatchedBasicBlock {
        basic_block: Box::new(CQLinBasicBlock),
      },
      BatchedBasicBlock {
        basic_block: Box::new(ReLUBasicBlock),
      },
      BatchedBasicBlock {
        basic_block: Box::new(CQ2BasicBlock { table_dict: HashMap::new() }),
      },
    ],
    nodes: vec![
      Node {
        basic_block: 0,
        inputs: vec![(-1, 0)],
      },
      Node {
        basic_block: 1,
        inputs: vec![(0, 0)],
      },
      Node {
        basic_block: 2,
        inputs: vec![(0, 0), (1, 0)],
      },
    ],
  };

  const m: usize = 1 << 4;
  const n: usize = 1 << 2;
  let matrix: Vec<_> = (0..n * m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-2..2))).collect();
  let matrix = ArrayD::from_shape_vec(vec![m, n], matrix).unwrap();
  let matrix = vec![&matrix];
  let input: Vec<_> = (0..n).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let input = ArrayD::from_shape_vec(vec![n, 1], input).unwrap();

  //Run:
  let inputs = vec![&input];
  let empty = vec![];
  let (id, relu_cq_table) = util::gen_cq_table(vec![&graph.basic_blocks[1].basic_block], -(1 << 5), 1 << 6);
  let (id, relu_cq_table) = (arr1(&id).into_dyn(), arr1(&relu_cq_table).into_dyn());
  let relu_cq_table = vec![&id, &relu_cq_table];
  let models = vec![&matrix, &empty, &relu_cq_table];
  let outputs = graph.run(&inputs, &models);
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();

  //Setup:
  let models: Vec<Vec<ArrayD<Data>>> = models.iter().map(|model| convert_to_data(srs, model)).collect();
  let models: Vec<Vec<&ArrayD<Data>>> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models: Vec<&Vec<&ArrayD<Data>>> = models.iter().map(|x| x).collect();
  let setups = graph.setup(srs, &models);
  //Converting to affine
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();

  //Prove:
  let inputs: Vec<ArrayD<Data>> = convert_to_data(srs, &inputs);
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<Data>>> = outputs.iter().map(|output| convert_to_data(srs, output)).collect();
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  //Converting to affine
  let proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    proofs.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();

  //Verify:
  let models: Vec<Vec<ArrayD<DataEnc>>> = models.iter().map(|model| (**model).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  let models: Vec<Vec<&ArrayD<DataEnc>>> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models: Vec<&Vec<&ArrayD<DataEnc>>> = models.iter().map(|x| x).collect();
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  let outputs: Vec<Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
