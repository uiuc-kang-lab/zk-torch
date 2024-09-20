#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
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
  // setups[1].0[0].serialize_uncompressed(File::create(&CONFIG.prover.setup_path).unwrap()).unwrap();
  setups.serialize_uncompressed(File::create(&CONFIG.prover.setup_path).unwrap()).unwrap();
  for i in setups {
    for j in i.0 {
      j.into_affine().check().unwrap();
    }
  }
  let modelsBytes = bincode::serialize(&models).unwrap();
  fs::write(&CONFIG.prover.model_path, &modelsBytes).unwrap();
}

fn run(inputs: &Vec<&ArrayD<Fr>>, graph: &Graph, models: &Vec<&ArrayD<Fr>>, timing: &mut TimingTree) -> Vec<Vec<ArrayD<Fr>>> {
  // Run:
  timed!(timing, "run witness generation", graph.run(inputs, models))
}

fn prove(srs: &SRS, inputs: &Vec<&ArrayD<Fr>>, outputs: Vec<Vec<ArrayD<Fr>>>, graph: &mut Graph, timing: &mut TimingTree) {
  // Load model and setup:
  println!("deserializing setups");
  let setups =
    Vec::<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>::deserialize_uncompressed(File::open(&CONFIG.prover.setup_path).unwrap())
      .unwrap();
  println!("finished deserializing setups");
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
  File::open(&CONFIG.prover.model_path).unwrap().read_to_end(&mut modelsBytes).unwrap();
  let models: Vec<ArrayD<Data>> = bincode::deserialize(&modelsBytes).unwrap();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();

  // Encode Data:
  let modelsEnc: Vec<ArrayD<DataEnc>> = util::vec_iter(&models).map(|model| (**model).map(|x| DataEnc::new(srs, x))).collect();
  let inputs: Vec<ArrayD<Data>> = timed!(
    timing,
    "encode inputs",
    util::vec_iter(inputs).map(|input| convert_to_data(srs, input)).collect()
  );
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let inputsEnc: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();
  let outputs = timed!(timing, "encode outputs", graph.encodeOutputs(srs, &models, &inputs, &outputs, timing));
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> =
    outputs.iter().map(|output| (**output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();

  // Save files:
  let modelsEncBytes = bincode::serialize(&modelsEnc).unwrap();
  let inputsEncBytes = bincode::serialize(&inputsEnc).unwrap();
  let outputsEncBytes = bincode::serialize(&outputsEnc).unwrap();
  fs::write(&CONFIG.prover.enc_model_path, &modelsEncBytes).unwrap();
  fs::write(&CONFIG.prover.enc_input_path, &inputsEncBytes).unwrap();
  fs::write(&CONFIG.prover.enc_output_path, &outputsEncBytes).unwrap();

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);

  // Prove:
  let proofs = timed!(timing, "prove", graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng, timing));
  proofs.serialize_uncompressed(File::create(&CONFIG.prover.proof_path).unwrap()).unwrap();
}

fn verify(srs: &SRS, graph: &Graph, timing: &mut TimingTree) {
  // Read Files:
  let proofs =
    Vec::<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>::deserialize_uncompressed_unchecked(File::open(&CONFIG.verifier.proof_path).unwrap()).unwrap();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
  let mut modelsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_model_path).unwrap().read_to_end(&mut modelsEncBytes).unwrap();
  let modelsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&modelsEncBytes).unwrap();
  let modelsEnc: Vec<&ArrayD<DataEnc>> = modelsEnc.iter().map(|model| model).collect();
  let mut inputsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_input_path).unwrap().read_to_end(&mut inputsEncBytes).unwrap();
  let inputsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&inputsEncBytes).unwrap();
  let inputsEnc: Vec<&ArrayD<DataEnc>> = inputsEnc.iter().map(|input| input).collect();
  let mut outputsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_output_path).unwrap().read_to_end(&mut outputsEncBytes).unwrap();
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
  #[cfg(feature = "debug")]
  timed!(
    timing,
    "verify (debug)",
    graph.verify_for_each_pairing(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng)
  );
  #[cfg(not(feature = "debug"))]
  timed!(
    timing,
    "verify",
    graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng)
  );
}

fn main() {
  // Timing
  let mut timing = TimingTree::default();
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  env_logger::init();

  let srs = &ptau::load_file(&CONFIG.ptau.ptau_path, CONFIG.ptau.pow_len_log, CONFIG.ptau.loaded_pow_len_log);
  let onnx_file_name = &CONFIG.onnx.model_path;
  let (mut graph, models) = onnx::load_file(onnx_file_name);
  let input_path = &CONFIG.onnx.input_path;
  let inputs = if std::path::Path::new(input_path).exists() {
    util::load_inputs_from_json_for_onnx(onnx_file_name, input_path)
  } else {
    util::generate_fake_inputs_for_onnx(onnx_file_name)
  };
  let inputs = inputs.iter().map(|x| x).collect();
  let models = models.iter().map(|x| &x.0).collect();
  setup(&srs, &graph, &models, &mut timing);
  let outputs = run(&inputs, &graph, &models, &mut timing);
  prove(&srs, &inputs, outputs, &mut graph, &mut timing);
  verify(&srs, &graph, &mut timing);

  // measure proof size
  measure_file_size(&CONFIG.prover.enc_model_path);
  measure_file_size(&CONFIG.prover.enc_input_path);
  measure_file_size(&CONFIG.prover.enc_output_path);
  measure_file_size(&CONFIG.prover.proof_path);
  timing.print();
  println!("Cargo run was successful.");
}
