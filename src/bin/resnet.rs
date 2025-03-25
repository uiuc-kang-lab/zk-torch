use ark_bn254::Fr;
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use std::env;
use zk_torch::graph::Graph;
use zk_torch::{onnx, util, CONFIG};

fn run(
  inputs: &Vec<&ArrayD<Fr>>,
  graph: &Graph,
  models: &Vec<&ArrayD<Fr>>,
  timing: &mut TimingTree,
) -> Result<Vec<Vec<ArrayD<Fr>>>, util::CQOutOfRangeError> {
  // Run:
  timed!(timing, "run witness generation", graph.run(inputs, models))
}
fn main() {
  // Timing
  let mut timing = TimingTree::default();
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  env_logger::init();

  // Collect command-line arguments
  let args: Vec<String> = env::args().collect();

  // Check if exactly two arguments were provided (excluding the program name)
  if args.len() != 4 {
    eprintln!("Usage: cargo run -- <integer1> <integer2>");
    return;
  }

  // Parse the arguments into integers
  let first_number: usize = args[2].parse().expect("Failed to parse the first argument as an integer");
  let second_number: usize = args[3].parse().expect("Failed to parse the second argument as an integer");

  println!("You entered: {} and {}", first_number, second_number);

  let onnx_file_name = &CONFIG.onnx.model_path;
  let (graph, models) = onnx::load_file(onnx_file_name);

  for i in first_number..second_number {
    // input_path is data/{idx}.json
    let input_path = format!("data/{}.json", i);
    let inputs = if std::path::Path::new(&input_path).exists() {
      util::load_inputs_from_json_for_onnx(onnx_file_name, &input_path)
    } else {
      util::generate_fake_inputs_for_onnx(onnx_file_name)
    };
    let inputs = inputs.iter().map(|x| x).collect();
    let models = models.iter().map(|x| &x.0).collect();
    let outputs = run(&inputs, &graph, &models, &mut timing);
    let output = &outputs.unwrap()[711][0].clone();
    //println!("output: {:?}", output.map(|x| util::fr_to_int(*x))); // shape [1, 1024]
    // find argmax
    // max = i128::MIN;
    let mut max = i128::MIN;
    let mut max_index = 0;
    for (i, x) in output.iter().enumerate() {
      let x = util::fr_to_int(*x);
      if x > max && i < 1000 {
        max = x;
        max_index = i;
      }
    }
    println!("=== data: {:?} | pred: {:?} ===", i, max_index);
  }

  timing.print();
  println!("Witness generation done");
}
