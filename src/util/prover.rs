/*
 * Prover utilities:
 * The functions are used for proving-related operations, such as
 * generating CQ tables and converting them to Data (generating commitment).
 */
use crate::basic_block::{BasicBlock, Data, DataEnc, SRS};
use crate::graph::Graph;
use crate::util::{measure_file_size, verify};
use crate::{onnx, ptau, util, CONFIG, LAYER_SETUP_DIR};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use ndarray::{arr0, arr1, concatenate, Array1, ArrayD, Axis, IxDyn};
use plonky2::{timed, util::timing::TimingTree};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use rayon::range;
use sha3::{Digest, Keccak256};
use std::fs::{self, File};
use std::io::Read;

#[derive(Debug, Clone, PartialEq)]
pub enum CQArrayType {
  Negative,
  NonNegative,
  NonZero,
  NonPositive,
  Positive,
  Custom(Vec<Fr>),
}

pub fn get_cq_N(cq_type: &CQArrayType) -> usize {
  match cq_type {
    CQArrayType::Negative => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::NonNegative => *onnx::CQ_RANGE as usize,
    CQArrayType::NonZero => (2 * (-*onnx::CQ_RANGE_LOWER) + 1) as usize,
    CQArrayType::NonPositive => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::Positive => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::Custom(range) => range.len(),
  }
}

pub fn gen_cq_array(cq_type: CQArrayType) -> ArrayD<Fr> {
  let r = match cq_type {
    CQArrayType::Negative => (*onnx::CQ_RANGE_LOWER..0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonNegative => (0..*onnx::CQ_RANGE as i32).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonZero => (*onnx::CQ_RANGE_LOWER..-*onnx::CQ_RANGE_LOWER + 1).filter(|&x| x != 0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonPositive => (*onnx::CQ_RANGE_LOWER + 1..1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Positive => (1..-*onnx::CQ_RANGE_LOWER + 1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Custom(range) => range,
  };
  arr1(&r).into_dyn()
}

pub fn check_cq_array(cq_type: CQArrayType, x_int: i128) -> bool {
  let result = match cq_type {
    CQArrayType::Negative => x_int < 0 && x_int >= (*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::NonNegative => x_int >= 0 && x_int < (*onnx::CQ_RANGE as i128),
    CQArrayType::NonZero => x_int != 0 && x_int >= (*onnx::CQ_RANGE_LOWER as i128) && x_int <= (-*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::NonPositive => x_int <= 0 && x_int > (*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::Positive => x_int > 0 && x_int <= (-*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::Custom(range) => {
      let range = range.iter().map(|x| util::fr_to_int(*x)).collect::<Vec<_>>();
      range.contains(&x_int)
    }
  };
  if !result {
    println!("{:?}", x_int);
  }
  result
}

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, offset: i128, size: usize) -> ArrayD<Fr> {
  let range = Array1::from_shape_fn(size, |i| Fr::from(i as u32) + Fr::from(offset)).into_dyn();
  let result = &(**basic_block).run(&ArrayD::zeros(IxDyn(&[0])), &vec![&range]).unwrap()[0];
  let range = range.view().into_shape(IxDyn(&[1, size])).unwrap();
  let result = result.view().into_shape(IxDyn(&[1, size])).unwrap();
  concatenate(Axis(0), &[range, result]).unwrap()
}

pub fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() <= 1 {
    return arr0(Data::new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  let mut a = a.map_axis(Axis(a.ndim() - 1), |r| Data {
    raw: r.as_standard_layout().as_slice().unwrap().to_vec(),
    poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
    r: Fr::zero(),
    g1: G1Projective::zero(),
  });
  a.par_map_inplace(|x| {
    *x = Data::new(srs, &x.raw);
  });
  a
}

pub fn convert_to_mock_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() <= 1 {
    return arr0(mock_data_new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  let mut a = a.map_axis(Axis(a.ndim() - 1), |r| Data {
    raw: r.as_standard_layout().as_slice().unwrap().to_vec(),
    poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
    r: Fr::zero(),
    g1: G1Projective::zero(),
  });
  a.par_map_inplace(|x| {
    *x = mock_data_new(srs, &x.raw);
  });
  a
}

pub fn mock_data_new(srs: &SRS, raw: &[Fr]) -> Data {
  let N = raw.len();
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  let f = DensePolynomial::from_coefficients_vec(domain.ifft(&raw));
  let fx = if f.is_zero() { G1Projective::zero() } else { srs.X1P[0].clone() };
  return Data {
    raw: raw.to_vec(),
    poly: f,
    g1: fx,
    r: Fr::from(1),
  };
}

pub fn witness_gen(
  inputs: &Vec<&ArrayD<Fr>>,
  graph: &Graph,
  models: &Vec<&ArrayD<Fr>>,
  timing: &mut TimingTree,
) -> Result<Vec<Vec<ArrayD<Fr>>>, util::CQOutOfRangeError> {
  // Run:
  timed!(timing, "run witness generation", graph.run(inputs, models))
}

pub fn prove(
  srs: &SRS,
  inputs: &Vec<&ArrayD<Fr>>,
  outputs: Vec<Vec<ArrayD<Fr>>>,
  setups: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)>,
  models: Vec<&ArrayD<Data>>,
  graph: &mut Graph,
  timing: &mut TimingTree,
) {
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
  #[cfg(feature = "fold")]
  let (proofs, acc_proofs) = timed!(timing, "prove", graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng, timing));
  #[cfg(not(feature = "fold"))]
  let proofs = timed!(timing, "prove", graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng, timing));

  proofs.serialize_uncompressed(File::create(&CONFIG.prover.proof_path).unwrap()).unwrap();
  #[cfg(feature = "fold")]
  acc_proofs.serialize_uncompressed(File::create(&CONFIG.prover.acc_proof_path).unwrap()).unwrap();
}

#[cfg(not(feature = "mock_prove"))]
pub fn setup(srs: &SRS, graph: &Graph, models: &Vec<&ArrayD<Fr>>, timing: &mut TimingTree) {
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

#[cfg(feature = "mock_prove")]
pub fn setup(
  srs: &SRS,
  graph: &Graph,
  models: &Vec<&ArrayD<Fr>>,
  timing: &mut TimingTree,
) -> (Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>, Vec<ArrayD<Data>>) {
  // Setup:
  let models: Vec<ArrayD<Data>> = models
    .par_iter()
    .map(|model| {
      let model = convert_to_mock_data(srs, model);
      model
    })
    .collect();

  let models_ref: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = timed!(timing, "setup and encode models", graph.setup(srs, &models_ref));
  (setups, models)
}

fn load_model() -> Vec<ArrayD<Data>> {
  let mut modelsBytes = Vec::new();
  File::open(&CONFIG.prover.model_path).unwrap().read_to_end(&mut modelsBytes).unwrap();
  let models: Vec<ArrayD<Data>> = bincode::deserialize(&modelsBytes).unwrap();
  models
}

pub fn zktorch_kernel() {
  // Timing
  let mut timing = TimingTree::default();
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
  let outputs = witness_gen(&inputs, &graph, &models, &mut timing);
  if outputs.is_err() {
    println!("CQ Error: {:?}", outputs.err().unwrap());
    return;
  }

  #[cfg(not(feature = "mock_prove"))]
  setup(&srs, &graph, &models, &mut timing);
  #[cfg(feature = "mock_prove")]
  let (setups, models) = setup(&srs, &graph, &models, &mut timing);

  // Load model and setup:
  #[cfg(not(feature = "mock_prove"))]
  let setups =
    Vec::<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>::deserialize_uncompressed(File::open(&CONFIG.prover.setup_path).unwrap())
      .unwrap();
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

  #[cfg(not(feature = "mock_prove"))]
  let models = load_model();
  let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();

  // Prove
  prove(&srs, &inputs, outputs.unwrap(), setups, models, &mut graph, &mut timing);

  // Verify
  verify(&srs, &graph, &mut timing);

  // Measure proof size;
  measure_file_size(&CONFIG.prover.enc_model_path);
  measure_file_size(&CONFIG.prover.enc_input_path);
  measure_file_size(&CONFIG.prover.enc_output_path);
  measure_file_size(&CONFIG.prover.proof_path);
  #[cfg(feature = "fold")]
  measure_file_size(&CONFIG.prover.final_proof_path);
  timing.print();
  println!("Cargo run was successful.");
}
