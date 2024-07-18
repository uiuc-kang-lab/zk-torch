/*
 * FFT utilities:
 * The functions are used for performing FFT and IFFT on G1 and G2 points.
 * Each function has a CPU and GPU implementation.
 */
#![allow(dead_code)]
#![allow(unused_imports)]
use ark_bn254::Fr;
use ark_ec::ScalarMul;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use {
  crate::util::{gpu_set_random_device, gpu_ssm_g1, gpu_ssm_g2},
  ark_bn254::{G1Projective, G2Projective},
  ark_ec::short_weierstrass::Projective,
  icicle_bn254::curve::{G1Projective as IG1P, G2Projective as IG2P, ScalarField},
  icicle_core::gfft,
  icicle_core::traits::ArkConvertible,
  icicle_cuda_runtime::memory::HostOrDeviceSlice,
};

fn bitreverse(mut n: u32, l: u64) -> u32 {
  let mut r = 0;
  for _ in 0..l {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  r
}

#[cfg(feature = "gpu")]
pub fn fft<G: GpuFft + std::clone::Clone>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, false);
  r
}

#[cfg(not(feature = "gpu"))]
pub fn fft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, false);
  r
}

#[cfg(feature = "gpu")]
pub fn ifft<G: GpuFft + std::clone::Clone>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, true);
  r
}

#[cfg(not(feature = "gpu"))]
pub fn ifft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, true);
  r.par_iter_mut().for_each(|x| *x *= domain.size_inv());
  r
}

#[cfg(feature = "gpu")]
pub fn fft_in_place<G: GpuFft + std::clone::Clone>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, false);
}

#[cfg(not(feature = "gpu"))]
pub fn fft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, false);
}

#[cfg(feature = "gpu")]
pub fn ifft_in_place<G: GpuFft + std::clone::Clone>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, true);
}

#[cfg(not(feature = "gpu"))]
pub fn ifft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, true);
  a.par_iter_mut().for_each(|x| *x *= domain.size_inv());
}

#[cfg(feature = "gpu")]
pub fn fft_helper<G: GpuFft + std::clone::Clone>(a: &mut Vec<G>, domain: GeneralEvaluationDomain<Fr>, inv: bool) {
  if inv {
    let gpu_result = G::gpu_ifft(domain, &a);
    *a = gpu_result;
  } else {
    let gpu_result = G::gpu_fft(domain, &a);
    *a = gpu_result;
  }
}

#[cfg(not(feature = "gpu"))]
pub fn fft_helper<G: ScalarMul + std::ops::MulAssign<Fr>>(a: &mut Vec<G>, domain: GeneralEvaluationDomain<Fr>, inv: bool) {
  let n = a.len();
  let log_size = domain.log_size_of_group();

  let swap = &mut Vec::new();
  (0..n).into_par_iter().map(|i| a[bitreverse(i as u32, log_size) as usize]).collect_into_vec(swap);

  let mut curr = (swap, a);

  let mut m = 1;
  for _ in 0..log_size {
    (0..n)
      .into_par_iter()
      .map(|i| {
        let left = i % (2 * m) < m;
        let k = i / (2 * m) * (2 * m);
        let j = i % m;
        let w = match inv {
          false => domain.element(n / (2 * m) * j),
          true => domain.element(n - n / (2 * m) * j),
        };
        let mut t = curr.0[(k + m) + j];
        t *= w;
        if left {
          return curr.0[k + j] + t;
        } else {
          return curr.0[k + j] - t;
        }
      })
      .collect_into_vec(curr.1);
    curr = (curr.1, curr.0);
    m *= 2;
  }
  if log_size % 2 == 0 {
    (0..n).into_par_iter().map(|i| curr.0[i]).collect_into_vec(curr.1);
  }
}

#[cfg(feature = "gpu")]
pub trait GpuFft {
  fn gpu_fft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self>
  where
    Self: Sized;

  fn gpu_ifft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self>
  where
    Self: Sized;
}

#[cfg(feature = "gpu")]
impl GpuFft for Projective<ark_bn254::g1::Config> {
  fn gpu_fft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self> {
    gpu_fft_g1(domain, a)
  }

  fn gpu_ifft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self> {
    gpu_ifft_g1(domain, a)
  }
}

#[cfg(feature = "gpu")]
impl GpuFft for Projective<ark_bn254::g2::Config> {
  fn gpu_fft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self> {
    gpu_fft_g2(domain, a)
  }

  fn gpu_ifft(domain: GeneralEvaluationDomain<Fr>, a: &Vec<Self>) -> Vec<Self> {
    gpu_ifft_g2(domain, a)
  }
}

#[cfg(feature = "gpu")]
pub fn gpu_fft_g1(domain: GeneralEvaluationDomain<Fr>, points: &Vec<G1Projective>) -> Vec<G1Projective> {
  gpu_set_random_device();
  gpu_fft_g1_helper(domain.group_gen(), points)
}

#[cfg(feature = "gpu")]
pub fn gpu_fft_g2(domain: GeneralEvaluationDomain<Fr>, points: &Vec<G2Projective>) -> Vec<G2Projective> {
  gpu_set_random_device();
  gpu_fft_g2_helper(domain.group_gen(), points)
}

#[cfg(feature = "gpu")]
pub fn gpu_ifft_g1(domain: GeneralEvaluationDomain<Fr>, points: &Vec<G1Projective>) -> Vec<G1Projective> {
  gpu_set_random_device();
  let points = gpu_fft_g1_helper(domain.group_gen_inv(), points);
  let scalars = vec![domain.size_inv(); points.len()];
  gpu_ssm_g1(&points, &scalars)
}

#[cfg(feature = "gpu")]
pub fn gpu_ifft_g2(domain: GeneralEvaluationDomain<Fr>, points: &Vec<G2Projective>) -> Vec<G2Projective> {
  gpu_set_random_device();
  let points = gpu_fft_g2_helper(domain.group_gen_inv(), points);
  let scalars = vec![domain.size_inv(); points.len()];
  gpu_ssm_g2(&points, &scalars)
}

#[cfg(feature = "gpu")]
pub fn gpu_fft_g1_helper(omega: Fr, points: &Vec<G1Projective>) -> Vec<G1Projective> {
  gpu_set_random_device();
  let size = points.len();
  let omega = vec![ScalarField::from_ark(omega)];
  let omega = HostOrDeviceSlice::on_host(omega);
  let points: Vec<_> = points.par_iter().map(|x| IG1P::from_ark(*x)).collect();
  let points = HostOrDeviceSlice::on_host(points);
  let results = vec![IG1P::zero(); size];
  let mut results: HostOrDeviceSlice<'_, IG1P> = HostOrDeviceSlice::on_host(results);
  let start = std::time::Instant::now();
  gfft::gfft(&omega, &points, &mut results).unwrap();
  println!("fft {size}: {:?}", start.elapsed().as_micros());
  results.as_slice().par_iter().map(|x| x.to_ark()).collect()
}

#[cfg(feature = "gpu")]
pub fn gpu_fft_g2_helper(omega: Fr, points: &Vec<G2Projective>) -> Vec<G2Projective> {
  gpu_set_random_device();
  let size = points.len();
  let omega = vec![ScalarField::from_ark(omega)];
  let omega = HostOrDeviceSlice::on_host(omega);
  let points: Vec<_> = points.par_iter().map(|x| IG2P::from_ark(*x)).collect();
  let points = HostOrDeviceSlice::on_host(points);
  let results = vec![IG2P::zero(); size];
  let mut results: HostOrDeviceSlice<'_, IG2P> = HostOrDeviceSlice::on_host(results);
  let start = std::time::Instant::now();
  gfft::gfft(&omega, &points, &mut results).unwrap();
  println!("fft2 {size}: {:?}", start.elapsed().as_micros());
  results.as_slice().par_iter().map(|x| x.to_ark()).collect()
}
