use zk_torch::util::zktorch_kernel;

fn main() {
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  zktorch_kernel();
}
