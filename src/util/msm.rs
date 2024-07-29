/*
 * MSM utilities:
 * The functions are used for performing MSM, SSM, and toeplitz mul on G1 and G2 points.
 * Each function has a CPU and GPU implementation.
 */
#![allow(unused_imports)]
use crate::util::{fft, ifft_in_place};
use ark_bn254::{Fr, G1Projective};
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::Zero;
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use {
  crate::util::{gpu_fft_g1, gpu_ifft_g1, gpu_set_random_device},
  ark_bn254::G2Projective,
  ark_ec::short_weierstrass::{Affine, Projective},
  icicle_bn254::curve::{G1Affine as IG1A, G1Projective as IG1P, G2Affine as IG2A, G2Projective as IG2P, ScalarField},
  icicle_core::gfft,
  icicle_core::traits::ArkConvertible,
  icicle_cuda_runtime::memory::HostOrDeviceSlice,
};

#[cfg(feature = "gpu")]
pub fn msm<G: GpuMsmProjective + std::clone::Clone + ark_ec::ScalarMul>(a: &[G::GpuMsmAffine], b: &[Fr]) -> G {
  G::gpu_msm(a, b)
}

#[cfg(not(feature = "gpu"))]
pub fn msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
  cpu_msm(a, b)
}

pub fn ssm_g1_in_place(points: &mut Vec<G1Projective>, scalars: &Vec<Fr>) {
  #[cfg(feature = "gpu")]
  {
    *points = gpu_ssm_g1(points, scalars);
  }

  #[cfg(not(feature = "gpu"))]
  {
    points.par_iter_mut().zip(scalars.par_iter()).for_each(|(x, scalar)| {
      *x *= *scalar;
    });
  }
}

#[cfg(feature = "gpu")]
pub fn circulant_mul(domain: GeneralEvaluationDomain<Fr>, c: &Vec<Fr>, a: &Vec<G1Projective>) -> Vec<G1Projective> {
  gpu_set_random_device();
  let lambda = domain.fft(c);
  let mut r = gpu_fft_g1(domain, a);
  r = gpu_ssm_g1(&r, &lambda);
  gpu_ifft_g1(domain, &r)
}

#[cfg(not(feature = "gpu"))]
pub fn circulant_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, c: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let lambda = domain.fft(c);
  let mut r = fft(domain, a);
  r.par_iter_mut().enumerate().for_each(|(i, x)| *x *= lambda[i]);
  ifft_in_place(domain, &mut r);
  r
}

pub fn toeplitz_mul(domain: GeneralEvaluationDomain<Fr>, m: &Vec<Fr>, a: &Vec<G1Projective>) -> Vec<G1Projective> {
  let n = (m.len() + 1) / 2;
  let mut temp = m.to_vec();
  let mut m2 = temp.split_off(n - 1);
  m2.push(Fr::zero());
  m2.append(&mut temp);
  let mut temp2 = a.to_vec();
  temp2.resize(2 * n, G1Projective::zero());
  let mut r = circulant_mul(domain, &m2, &temp2);
  r.resize(n, G1Projective::zero());
  r
}

fn cpu_msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
  let max_threads = rayon::current_num_threads();
  let size = ark_std::cmp::min(a.len(), b.len());
  if max_threads > size {
    return VariableBaseMSM::msm_unchecked(&a, &b);
  }
  let chunk_size = size / max_threads;
  let a = &a[..size];
  let b = &b[..size];
  let a_chunks = a.par_chunks(chunk_size);
  let b_chunks = b.par_chunks(chunk_size);
  return a_chunks.zip(b_chunks).map(|(x, y)| -> P { VariableBaseMSM::msm_unchecked(&x, &y) }).sum();
}

#[cfg(feature = "gpu")]
pub trait GpuMsmProjective {
  type GpuMsmAffine;
  fn gpu_msm(a: &[Self::GpuMsmAffine], b: &[Fr]) -> Self
  where
    Self: Sized;
}

#[cfg(feature = "gpu")]
impl GpuMsmProjective for Projective<ark_bn254::g1::Config> {
  type GpuMsmAffine = Affine<ark_bn254::g1::Config>;
  fn gpu_msm(a: &[Self::GpuMsmAffine], b: &[Fr]) -> Self {
    let a: Vec<_> = a.par_iter().map(|x| IG1A::from_ark(*x)).collect();
    let b: Vec<_> = b.par_iter().map(|x| *x).collect();
    gpu_msm_g1(&a, &b)
  }
}

#[cfg(feature = "gpu")]
impl GpuMsmProjective for Projective<ark_bn254::g2::Config> {
  type GpuMsmAffine = Affine<ark_bn254::g2::Config>;
  fn gpu_msm(a: &[Self::GpuMsmAffine], b: &[Fr]) -> Self {
    let a: Vec<_> = a.par_iter().map(|x| IG2A::from_ark(*x)).collect();
    let b: Vec<_> = b.par_iter().map(|x| *x).collect();
    gpu_msm_g2(&a, &b)
  }
}

#[cfg(feature = "gpu")]
pub fn gpu_msm_g1(points: &Vec<IG1A>, scalars: &Vec<Fr>) -> G1Projective {
  gpu_set_random_device();
  let size = ark_std::cmp::min(points.len(), scalars.len());
  if size < 32 {
    let points: Vec<_> = points.par_iter().map(|x| x.to_ark()).collect();
    return cpu_msm(&points, scalars);
  }
  let cfg = icicle_core::msm::MSMConfig::default();
  let points = HostOrDeviceSlice::on_host(points[..size].to_vec());
  let scalars = scalars[..size].par_iter().map(|x| ScalarField::from_ark(*x)).collect();
  let scalars = HostOrDeviceSlice::on_host(scalars);
  let results = vec![IG1P::zero(); 1];
  let mut results: HostOrDeviceSlice<'_, IG1P> = HostOrDeviceSlice::on_host(results);
  icicle_core::msm::msm(&scalars, &points, &cfg, &mut results).unwrap();
  results.as_slice()[0].to_ark()
}

#[cfg(feature = "gpu")]
pub fn gpu_msm_g2(points: &Vec<IG2A>, scalars: &Vec<Fr>) -> G2Projective {
  gpu_set_random_device();
  let size = ark_std::cmp::min(points.len(), scalars.len());
  if size < 32 {
    let points: Vec<_> = points.iter().map(|x| x.to_ark()).collect();
    return cpu_msm(&points, scalars);
  }
  let cfg = icicle_core::msm::MSMConfig::default();
  let points = HostOrDeviceSlice::on_host(points[..size].to_vec());
  let scalars = scalars[..size].par_iter().map(|x| ScalarField::from_ark(*x)).collect();
  let scalars = HostOrDeviceSlice::on_host(scalars);
  let results = vec![IG2P::zero(); 1];
  let mut results: HostOrDeviceSlice<'_, IG2P> = HostOrDeviceSlice::on_host(results);
  icicle_core::msm::msm(&scalars, &points, &cfg, &mut results).unwrap();
  results.as_slice()[0].to_ark()
}

#[cfg(feature = "gpu")]
pub fn gpu_ssm_g1(points: &Vec<G1Projective>, scalars: &Vec<Fr>) -> Vec<G1Projective> {
  gpu_set_random_device();
  let size = points.len();
  let points: Vec<_> = points.par_iter().map(|x| IG1P::from_ark(*x)).collect();
  let points = HostOrDeviceSlice::on_host(points);
  let scalars = scalars.par_iter().map(|x| ScalarField::from_ark(*x)).collect();
  let scalars = HostOrDeviceSlice::on_host(scalars);
  let results = vec![IG1P::zero(); size];
  let mut results: HostOrDeviceSlice<'_, IG1P> = HostOrDeviceSlice::on_host(results);
  let start = std::time::Instant::now();
  gfft::ssm(&scalars, &points, &mut results).unwrap();
  println!("ssm {size}: {:?}", start.elapsed().as_micros());
  results.as_slice().par_iter().map(|x| x.to_ark()).collect()
}

#[cfg(feature = "gpu")]
pub fn gpu_ssm_g2(points: &Vec<G2Projective>, scalars: &Vec<Fr>) -> Vec<G2Projective> {
  gpu_set_random_device();
  let size = points.len();
  let points: Vec<_> = points.par_iter().map(|x| IG2P::from_ark(*x)).collect();
  let points = HostOrDeviceSlice::on_host(points);
  let scalars = scalars.par_iter().map(|x| ScalarField::from_ark(*x)).collect();
  let scalars = HostOrDeviceSlice::on_host(scalars);
  let results = vec![IG2P::zero(); size];
  let mut results: HostOrDeviceSlice<'_, IG2P> = HostOrDeviceSlice::on_host(results);
  let start = std::time::Instant::now();
  gfft::ssm(&scalars, &points, &mut results).unwrap();
  println!("ssm2 {size}: {:?}", start.elapsed().as_micros());
  results.as_slice().par_iter().map(|x| x.to_ark()).collect()
}
