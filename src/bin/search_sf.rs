use core::panic;
use zk_torch::{onnx, util, CONFIG};

// Run the witness_gen binary with a new sf and return the std output
fn witness_gen(onnx_file_name: &str) -> String {
  let (graph, models) = onnx::load_file(onnx_file_name);

  let input_path = &CONFIG.onnx.input_path;
  let inputs = if std::path::Path::new(input_path).exists() {
    util::load_inputs_from_json_for_onnx(onnx_file_name, input_path)
  } else {
    util::generate_fake_inputs_for_onnx(onnx_file_name)
  };
  let inputs = inputs.iter().map(|x| x).collect();
  let models = models.iter().map(|x| &x.0).collect();
  let outputs = graph.run(&inputs, &models);
  if outputs.is_err() {
    "CQ Error".to_string()
  } else {
    "Success".to_string()
  }
}

fn update_sf(new_sf_log: usize) {
  let mut sf_log = onnx::SF_LOG.write().unwrap();
  *sf_log = new_sf_log;
  drop(sf_log);
  let mut sf = onnx::SF.write().unwrap();
  *sf = 1 << new_sf_log;
  drop(sf);
  let mut sf_float = onnx::SF_FLOAT.write().unwrap();
  *sf_float = (1 << new_sf_log) as f32;
  drop(sf_float);
}

// Given the CQ range, search for the optimal scale factor for the given model
fn search_optimal_sf(onnx_file_name: &str, cq_range_log: usize) -> usize {
  let loaded_pow_len_log = CONFIG.ptau.loaded_pow_len_log;
  assert!(cq_range_log < loaded_pow_len_log);
  let mut min_sf = 0;
  let mut max_sf = cq_range_log - 1;
  let mut current_sf = 0;
  let mut opt_sf = 0;
  let mut prev_sfs: Vec<usize> = Vec::new();
  // Binary search for the optimal scale factor
  // In each iteration, we try with the new scale factor, which is
  // the average of the min and max scale factors.
  while min_sf <= max_sf && prev_sfs.iter().find(|&&x| x == current_sf).is_none() {
    println!("==> Trying scale factor: 2^{}", current_sf);

    // Update the global scale factor by the new scale factor
    update_sf(current_sf);

    let stdout = witness_gen(onnx_file_name);

    // Check if the std output contains success message
    if stdout.contains("Success") {
      // If the output contains "Success", then the
      // optimal scale factor may be larger than the current scale factor
      // Set the minimum scale factor to the current scale factor
      min_sf = current_sf;
      opt_sf = current_sf;
    } else {
      // If the output does not contain "Success", then the current scale factor too high
      // Set the maximum scale factor to the current scale factor
      if current_sf == 0 {
        // If the current scale factor is 0, then the CQ range is too small for the given circuit
        panic!("CQ range is too small for the given circuit");
      }
      max_sf = current_sf;
    }
    prev_sfs.push(current_sf);
    current_sf = ((min_sf + max_sf) as f64 / 2.0).round() as usize;
  }
  opt_sf
}

fn main() {
  let cq_range_log = CONFIG.sf.cq_range_log;
  let onnx_file_name = &CONFIG.onnx.model_path;
  let optimal_sf = search_optimal_sf(onnx_file_name, cq_range_log);
  println!("==> Given the CQ range, the optimal scale factor for this model is 2^{}", optimal_sf);
  println!("==> Please set 'scale_factor_log={}' in the config file", optimal_sf);
}
