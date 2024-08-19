/*
 * Arithmetic utilities:
 * The functions are used for converting between Fr and i32, for calculating powers of Fr,
 * and for pointwise operations on u32 or f32.
 */
use ark_bn254::Fr;
use ark_ff::PrimeField;

pub fn fr_to_int(x: Fr) -> i32 {
  if x < Fr::from(1 << 28) {
    x.into_bigint().0[0] as i32
  } else {
    -((-x).into_bigint().0[0] as i32)
  }
}

pub fn calc_pow(alpha: Fr, n: usize) -> Vec<Fr> {
  let mut pow: Vec<Fr> = vec![alpha; n];
  if n > 0 {
    for i in 0..n - 1 {
      pow[i + 1] = pow[i] * alpha;
    }
  }
  pow
}

pub fn next_pow(n: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  let mut v = n;
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v += 1;
  v
}

/// Computes erf(x) approximation using A&S formula 7.1.26
pub fn erf(x: f32) -> f32 {
  let a1 = 0.254829592;
  let a2 = -0.284496736;
  let a3 = 1.421413741;
  let a4 = -1.453152027;
  let a5 = 1.061405429;
  let p = 0.3275911;
  let sign = if x < 0.0 { -1.0 } else { 1.0 };
  let x = x.abs();
  let t = 1.0 / (1.0 + p * x);
  let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
  sign * y
}
