#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::{Graph, Node};
use ndarray::{arr0, arr1, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
mod basic_block;
mod graph;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() == 0 {
    return arr0(Data::new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  a.map_axis(Axis(a.ndim() - 1), |r| Data::new(srs, r.as_slice().unwrap()))
}

fn main() {
  let srs = &ptau::load_file("challenge14", 14);
  let mut graph = Graph {
    basic_blocks: vec![Box::new(CQLinBasicBlock {})],
    nodes: vec![Node {
      basic_block: 0,
      inputs: vec![(-1, 0)],
    }],
  };

  const m: usize = 1 << 1;
  const n: usize = 1 << 2;
  const k: usize = 1 << 3;
  let input1: Vec<_> = (0..n * m).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let input1 = ArrayD::from_shape_vec(vec![n, m], input1).unwrap();
  let model: Vec<_> = (0..m * k).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();
  let model = ArrayD::from_shape_vec(vec![m, k], model).unwrap();

  //Run:
  let inputs = vec![&input1]; //, &input2
  let models = vec![&model];
  let outputs = graph.run(&inputs, &models);
  println!("{:?}", inputs.iter().map(|input| input.map(|x| util::fr_to_int(*x))).collect::<Vec<_>>());
  println!("{:?}", outputs[0][0].map(|x| util::fr_to_int(*x)));
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();

  //Setup:
  let models: Vec<ArrayD<Data>> = models.iter().map(|model| convert_to_data(srs, model)).collect();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = graph.setup(srs, &models);
  //Converting to affine
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();

  //Prove:
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|output| convert_to_data(srs, output)).collect()).collect();
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  //Converting to affine
  let proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    proofs.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();

  //Verify:
  let models: Vec<ArrayD<DataEnc>> = models.iter().map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let models: Vec<&ArrayD<DataEnc>> = models.iter().map(|model| model).collect();
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  let outputs: Vec<Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<DataEnc>>> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
