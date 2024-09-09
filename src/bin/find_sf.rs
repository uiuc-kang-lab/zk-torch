use core::panic;
use std::fs;
use std::process::{Command, Stdio};
use zk_torch::{CONFIG, CONFIG_FILE};

fn witness_gen() -> String {
  let output = Command::new("./target/release/witness_gen")
    .arg(CONFIG_FILE.clone())
    .stdout(Stdio::piped()) // Capture stdout
    .stderr(Stdio::piped()) // Capture stderr
    .output() // Run the command and capture the output
    .expect("Failed to execute command");

  // Capture stdout
  let stdout = String::from_utf8_lossy(&output.stdout);

  // Capture stderr (where panic messages typically appear)
  let stderr = String::from_utf8_lossy(&output.stderr);
  if !stderr.is_empty() {
    println!("stdout msg:\n{}", stdout);
    println!("stderr msg:\n{}", stderr);
  }
  format!("{}", stdout)
}

fn modify_config_sf(current_sf: usize, file_path: &str) {
  // Modify the config file with the new scale factor
  let mut config = (*CONFIG).clone();
  config.sf.scale_factor_log = current_sf;

  // Serialize the modified struct back into YAML
  let modified_yaml = serde_yaml::to_string(&config).unwrap();

  // Write the modified YAML back to the file
  let _ = fs::write(file_path, modified_yaml).unwrap();
}

// To run this, you need to have the witness_gen binary built
// Run `cargo build --release` in the zk-torch directory
// This will build the binary in zk-torch/target/release/witness_gen
fn main() {
  let file_path = &CONFIG_FILE.clone();
  let cq_range_log = CONFIG.sf.cq_range_log;
  let loaded_pow_len_log = CONFIG.ptau.loaded_pow_len_log;
  assert!(cq_range_log < loaded_pow_len_log);
  let mut min_sf = 0;
  let mut max_sf = cq_range_log - 1;
  let mut current_sf = 0;
  let mut opt_sf = 0;
  let mut prev_sfs: Vec<usize> = Vec::new();

  // Binary search for the optimal scale factor
  while min_sf <= max_sf && prev_sfs.iter().find(|&&x| x == current_sf).is_none() {
    println!("Trying scale factor: 2^{}", current_sf);

    modify_config_sf(current_sf, file_path);

    let stdout = witness_gen();

    // Check if the output contains the string "Success"
    if stdout.contains("Witness generation done") {
      // If the output contains "Witness generation done", then the scale factor is too high
      // Set the maximum scale factor to the current scale factor - 1
      min_sf = current_sf;
      opt_sf = current_sf;
    } else {
      // If the output does not contain "Success", then the scale factor is too low
      // Set the minimum scale factor to the current scale factor + 1
      max_sf = current_sf;
      if current_sf == 0 {
        panic!("CQ range is too small for the given circuit");
      }
    }
    prev_sfs.push(current_sf);
    current_sf = ((min_sf + max_sf) as f64 / 2.0).round() as usize;
  }
  println!("Optimal scale factor: 2^{}", opt_sf);
  modify_config_sf(opt_sf, file_path);
}
