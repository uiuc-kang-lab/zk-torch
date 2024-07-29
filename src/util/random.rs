/*
 * Random utilities:
 * The functions are used for adding randomness to the RNG and
 * setting the random device for GPU computations.
 */
#![allow(unused_imports)]
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use sha3::{Digest, Keccak256};

pub fn add_randomness(rng: &mut StdRng, mut bytes: Vec<u8>) {
  let mut buf = vec![0u8; 32];
  rng.fill_bytes(&mut buf);
  bytes.append(&mut buf);
  let mut buf = [0u8; 32];
  let mut hasher = Keccak256::new();
  hasher.update(bytes);
  hasher.finalize_into((&mut buf).into());
  *rng = StdRng::from_seed(buf);
}

#[cfg(feature = "gpu")]
pub fn gpu_set_random_device() {
  let mut rng = StdRng::from_entropy();
  icicle_cuda_runtime::device::set_device(rng.gen_range(0..1)).unwrap();
}
