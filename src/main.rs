#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use crate::graph::Graph;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use basic_block::*;
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};
use std::fs::{self, File};
use std::io::Read;
use util::{convert_to_data, measure_file_size};
mod basic_block;
mod graph;
mod layer;
mod onnx;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

macro_rules! stat_println {
  ($($arg:tt)*) => {
      println!("============> {}", format_args!($($arg)*));
  };
}

fn setup(srs: &SRS, graph: &Graph, models: &Vec<&ArrayD<Fr>>) {
  let start = std::time::Instant::now();
  // Setup:
  let models: Vec<ArrayD<Data>> = models.par_iter().map(|model| convert_to_data(srs, model)).collect();
  let models_ref: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = graph.setup(srs, &models_ref);
  stat_println!("Time taken to setup and encode models: {:?}", start.elapsed());
  // Save files:
  setups.serialize_uncompressed(File::create("setups").unwrap()).unwrap();
  let modelsBytes = bincode::serialize(&models).unwrap();
  fs::write("models", &modelsBytes).unwrap();
}

fn run(inputs: &Vec<&ArrayD<Fr>>, graph: &Graph, models: &Vec<&ArrayD<Fr>>) -> Vec<Vec<ArrayD<Fr>>> {
  let start = std::time::Instant::now();
  // Run:
  let outputs = graph.run(inputs, models);
  stat_println!("Time taken to run: {:?}", start.elapsed());
  outputs
}

fn prove(srs: &SRS, inputs: &Vec<&ArrayD<Fr>>, outputs: Vec<Vec<ArrayD<Fr>>>, graph: &mut Graph) {
  // Load model and setup:
  let setups = Vec::<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>::deserialize_uncompressed(File::open("setups").unwrap()).unwrap();
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> = util::vec_iter(&setups)
    .map(|x| {
      (
        util::vec_iter(&x.0).map(|y| (*y).into()).collect(),
        util::vec_iter(&x.1).map(|y| (*y).into()).collect(),
        util::vec_iter(&x.2).map(|y| (y.clone())).collect(),
      )
    })
    .collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
  
  let mut modelsBytes = Vec::new();
  File::open("models").unwrap().read_to_end(&mut modelsBytes).unwrap();
  let models: Vec<ArrayD<Data>> = bincode::deserialize(&modelsBytes).unwrap();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();

  let start = std::time::Instant::now();
  // Encode Data:
  let modelsEnc: Vec<ArrayD<DataEnc>> = util::vec_iter(&models).map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let inputs: Vec<ArrayD<Data>> = util::vec_iter(inputs).map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let inputsEnc: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();
  let outputs: Vec<Vec<ArrayD<Data>>> = graph.encodeOutputs(srs, &models, &inputs, &outputs);
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();
  let encode_time = start.elapsed();
  stat_println!("Time taken to encode I/O data: {:?}", encode_time);

  // Save files:
  let modelsEncBytes = bincode::serialize(&modelsEnc).unwrap();
  let inputsEncBytes = bincode::serialize(&inputsEnc).unwrap();
  let outputsEncBytes = bincode::serialize(&outputsEnc).unwrap();
  fs::write("modelsEnc", &modelsEncBytes).unwrap();
  fs::write("inputsEnc", &inputsEncBytes).unwrap();
  fs::write("outputsEnc", &outputsEncBytes).unwrap();

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);

  // Prove:
  let start = std::time::Instant::now();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  let prove_time = start.elapsed();
  stat_println!("Time taken to prove: {:?}", prove_time);
  stat_println!("Total time taken to prove: {:?}", encode_time + prove_time);
  proofs.serialize_uncompressed(File::create("proofs").unwrap()).unwrap();
}

fn verify(srs: &SRS, graph: &Graph) {
  // Read Files:
  let proofs = Vec::<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>::deserialize_uncompressed_unchecked(File::open("proofs").unwrap()).unwrap();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
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

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);

  // Verify:
  let start = std::time::Instant::now();
  #[cfg(feature = "debug")]
  graph.verify_for_each_pairing(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng);
  #[cfg(not(feature = "debug"))]
  graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng);
  stat_println!("Time taken to verify: {:?}", start.elapsed());
}

fn main() {
  let srs = &ptau::load_file("challenge", 7, 7);
  let onnx_file_name = "sample.onnx";
  let (mut graph, models) = onnx::load_file(onnx_file_name);
  let fake_inputs = util::generate_fake_inputs_for_onnx(onnx_file_name);
  let inputs = fake_inputs.iter().map(|x| x).collect();
  let models = models.iter().map(|x| x).collect();
  setup(&srs, &graph, &models);
  let outputs = run(&inputs, &graph, &models);
  prove(&srs, &inputs, outputs, &mut graph);
  verify(&srs, &graph);
  // measure_proof_size
  let _modelsEncSize = measure_file_size("modelsEnc");
  let inputsEncSize = measure_file_size("inputsEnc");
  let outputsEncSize = measure_file_size("outputsEnc");
  let proofsSize = measure_file_size("proofs");
  let totalSize = inputsEncSize + outputsEncSize + proofsSize;
  stat_println!("Total proof size: {}", util::format_file_size(totalSize));
  println!("Cargo run was successful.");
}
