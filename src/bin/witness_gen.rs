use ark_bn254::Fr;
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use zk_torch::graph::Graph;
use zk_torch::{onnx, util, CONFIG};

fn run(inputs: &Vec<&ArrayD<Fr>>, graph: &Graph, models: &Vec<&ArrayD<Fr>>, timing: &mut TimingTree) -> Vec<Vec<ArrayD<Fr>>> {
  // Run:
  timed!(timing, "run witness generation", graph.run(inputs, models))
}
fn main() {
  // Timing
  let mut timing = TimingTree::default();
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  env_logger::init();

  let onnx_file_name = &CONFIG.onnx.model_path;
  let (graph, models) = onnx::load_file(onnx_file_name);

  let input_path = &CONFIG.onnx.input_path;
  let inputs = if std::path::Path::new(input_path).exists() {
    util::load_inputs_from_json_for_onnx(onnx_file_name, input_path)
  } else {
    util::generate_fake_inputs_for_onnx(onnx_file_name)
  };
  let inputs = inputs.iter().map(|x| x).collect();
  let models = models.iter().map(|x| &x.0).collect();
  let _outputs = run(&inputs, &graph, &models, &mut timing);

  timing.print();
  println!("Witness generation done");
}
