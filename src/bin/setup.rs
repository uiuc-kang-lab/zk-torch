#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};
use std::fs::{self, File};
use std::io::Read;
use zk_torch::basic_block::*;
use zk_torch::graph::Graph;
use zk_torch::util::{self, convert_to_data, measure_file_size};
use zk_torch::{onnx, ptau, CONFIG, LAYER_SETUP_DIR};

fn setup(srs: &SRS, graph: &Graph, models: &Vec<&ArrayD<Fr>>, timing: &mut TimingTree) {
  // Setup:
  let models: Vec<ArrayD<Data>> = models
    .par_iter()
    .enumerate()
    .map(|(i, model)| {
      let bb = &graph.basic_blocks[i];
      let bb_name = format!("{bb:?}");
      let file_name = format!("{}.model", util::hash_str(&format!("{bb_name:?}")));
      let file_path = format!("{}/{}", *LAYER_SETUP_DIR, file_name);
      if util::file_exists(&file_path) {
        println!("CQs: Loading layer model from file: {}", file_path);
        let mut modelBytes = Vec::new();
        File::open(file_path).unwrap().read_to_end(&mut modelBytes).unwrap();
        let model: ArrayD<Data> = bincode::deserialize(&modelBytes).unwrap();
        model
      } else {
        let model = convert_to_data(srs, model);
        if bb_name.contains("CQ2BasicBlock") || bb_name.contains("CQBasicBlock") {
          let modelBytes = bincode::serialize(&model).unwrap();
          fs::write(file_path, &modelBytes).unwrap();
        }
        model
      }
    })
    .collect();

  let models_ref: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = timed!(timing, "setup and encode models", graph.setup(srs, &models_ref));
  // Save files:
  setups.serialize_uncompressed(File::create(&CONFIG.prover.setup_path).unwrap()).unwrap();
  let modelsBytes = bincode::serialize(&models).unwrap();
  fs::write(&CONFIG.prover.model_path, &modelsBytes).unwrap();
}

fn main() {
  // Timing
  let mut timing = TimingTree::default();
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  env_logger::init();

  let srs = &ptau::load_file(&CONFIG.ptau.ptau_path, CONFIG.ptau.pow_len_log, CONFIG.ptau.loaded_pow_len_log);
  let onnx_file_name = &CONFIG.onnx.model_path;
  let (graph, models) = onnx::load_file(onnx_file_name);
  let models = models.iter().map(|x| &x.0).collect();
  setup(&srs, &graph, &models, &mut timing);
  println!("Setup was successful.");
}
