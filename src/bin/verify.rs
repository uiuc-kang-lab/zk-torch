//! An example of a verifier binary
use plonky2::{timed, util::timing::TimingTree};
use zk_torch::{onnx, ptau, util, CONFIG, basic_block::SRS};

fn main() {
    // Timing
    let mut timing = TimingTree::default();
    // please export RUST_LOG=debug; the debug logs for timing will be printed
    env_logger::init();

    // Load SRS
    let srs = timed!(timing, "load SRS", ptau::load_file(&CONFIG.ptau.ptau_path, CONFIG.ptau.pow_len_log, CONFIG.ptau.loaded_pow_len_log));

    // Load graph
    let onnx_file_name = &CONFIG.onnx.model_path;
    let (graph, _) = onnx::load_file(onnx_file_name);

    // Verify
    util::verify(&srs, &graph, &mut timing);

    timing.print();
    println!("✅ Verification successful");
} 