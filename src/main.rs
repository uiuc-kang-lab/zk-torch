#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use crate::graph::Graph;
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use basic_block::*;
use ndarray::ArrayD;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};
use std::fs::{self, File};
use std::io::Read;
use util::convert_to_data;
mod basic_block;
mod graph;
mod layer;
mod onnx;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn prove(srs: &SRS, inputs: &Vec<&ArrayD<Fr>>, graph: &mut Graph, models: &Vec<&ArrayD<Fr>>) {
  // Run:
  let mut start = std::time::Instant::now();
  let outputs = graph.run(inputs, models);
  println!("run: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Setup:
  let models: Vec<ArrayD<Data>> = models
    .par_iter()
    .enumerate()
    .map(|(i, model)| {
      println!("encode model {:?} {:?}", i, model.shape());
      convert_to_data(srs, model)
    })
    .collect();
  println!("encode models: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = graph.setup(srs, &models);
  println!("setup: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Encode Data:
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.par_iter().map(|x| (x.0.par_iter().map(|y| (*y).into()).collect(), x.1.par_iter().map(|y| (*y).into()).collect())).collect();
  println!("setups to affine: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();
  let modelsEnc: Vec<ArrayD<DataEnc>> = models.par_iter().map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let inputs: Vec<ArrayD<Data>> = inputs.par_iter().map(|input| convert_to_data(srs, input)).collect();
  println!("encode inputs: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let inputsEnc: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();
  let outputs: Vec<Vec<ArrayD<Data>>> = graph.encodeOutputs(srs, &models, &inputs, &outputs);
  println!("encode outputs: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  println!("finishing up pointers: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Save files:
  let modelsEncBytes = bincode::serialize(&modelsEnc).unwrap();
  let inputsEncBytes = bincode::serialize(&inputsEnc).unwrap();
  let outputsEncBytes = bincode::serialize(&outputsEnc).unwrap();
  fs::write("modelsEnc", &modelsEncBytes).unwrap();
  fs::write("inputsEnc", &inputsEncBytes).unwrap();
  fs::write("outputsEnc", &outputsEncBytes).unwrap();
  println!("wrote files: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);
  println!("hashed: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Prove:
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  println!("prove: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
  proofs.serialize_uncompressed(File::create("proofs").unwrap()).unwrap();
  println!("wrote proof: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
}

fn verify(srs: &SRS, graph: &Graph) {
  let mut start = std::time::Instant::now();
  // Read Files:
  let proofs = Vec::<(Vec<G1Affine>, Vec<G2Affine>)>::deserialize_uncompressed_unchecked(File::open("proofs").unwrap()).unwrap();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();
  let mut modelsEncBytes = Vec::new();
  File::open("modelsEnc").unwrap().read_to_end(&mut modelsEncBytes).unwrap();
  let modelsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&modelsEncBytes).unwrap();
  let modelsEnc: Vec<&ArrayD<DataEnc>> = modelsEnc.iter().map(|model| model).collect();
  let mut inputsEncBytes = Vec::new();
  File::open("inputsEnc").unwrap().read_to_end(&mut inputsEncBytes).unwrap();
  let inputsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&inputsEncBytes).unwrap();
  let inputsEnc: Vec<&ArrayD<DataEnc>> = inputsEnc.iter().map(|input| input).collect();
  let mut outputsEncBytes = Vec::new();
  File::open("outputsEnc").unwrap().read_to_end(&mut outputsEncBytes).unwrap();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> = bincode::deserialize(&outputsEncBytes).unwrap();
  let outputsEnc: Vec<Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputsEnc: Vec<&Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|x| x).collect();
  println!("read files: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);
  println!("hashed: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();

  // Verify:
  graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng);
  println!("verify: {:?}",start.elapsed().as_micros()); start = std::time::Instant::now();
}

fn main() {
  let srs = &ptau::load_file("/home/arigf2/project/challenge_0085", 28, 26);
  let (mut graph, models) = onnx::load_file("distilbert_Opset16.onnx");
  let mut rng = StdRng::from_entropy();
  let input1: Vec<Fr> = (0..128).map(|_| Fr::from(rng.gen_range(0..30522))).collect();
  let input1 = ArrayD::from_shape_vec(vec![1,128], input1).unwrap();
  let input2: Vec<Fr> = (0..128).map(|_| Fr::from(rng.gen_range(0..2))).collect();
  let input2 = ArrayD::from_shape_vec(vec![1,128], input2).unwrap();
  let inputs = vec![&input1, &input2];
  let models = models.iter().map(|x| x).collect();
  prove(&srs, &inputs, &mut graph, &models);
  verify(&srs, &graph);
}
