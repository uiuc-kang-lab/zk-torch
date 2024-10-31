use zk_torch::util::gpu_set_random_device;
use zk_torch::util::zktorch_kernel;
fn main() {
  #[cfg(feature = "gpu")]
  gpu_set_random_device();
  // please export RUST_LOG=debug; the debug logs for timing will be printed
  zktorch_kernel();
}
