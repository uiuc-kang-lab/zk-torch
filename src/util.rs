#![allow(dead_code)]
use crate::BasicBlock;
use ark_bn254::Fr;
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::Zero;
use ndarray::{arr1, azip, ArrayD};
use rayon::prelude::*;

fn bitreverse(mut n: u32, l: u64) -> u32 {
  let mut r = 0;
  for _ in 0..l {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  r
}
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
pub fn fft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, false);
  r
}
pub fn ifft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  let mut r = a.to_vec();
  fft_helper(&mut r, domain, true);
  r.par_iter_mut().for_each(|x| *x *= domain.size_inv());
  r
}
pub fn fft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, false);
}
pub fn ifft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  fft_helper(a, domain, true);
  a.par_iter_mut().for_each(|x| *x *= domain.size_inv());
}

pub fn circulant_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, c: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let lambda = domain.fft(c);
  let mut r = fft(domain, a);
  r.par_iter_mut().enumerate().for_each(|(i, x)| *x *= lambda[i]);
  ifft_in_place(domain, &mut r);
  r
}

pub fn toeplitz_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, m: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let n = (m.len() + 1) / 2;
  let mut temp = m.to_vec();
  let mut m2 = temp.split_off(n - 1);
  m2.push(Fr::zero());
  m2.append(&mut temp);
  let mut temp2 = a.to_vec();
  temp2.resize(2 * n, G::zero());
  let mut r = circulant_mul(domain, &m2, &temp2);
  r.resize(n, G::zero());
  r
}

pub fn msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
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

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, table_size: usize) -> ArrayD<Fr> {
  let range: Vec<_> = (0..table_size).map(|i| Fr::from(i as u32) - Fr::from((table_size >> 1) as u32)).collect();
  let range = arr1(&range).into_dyn();
  let result = (*basic_block).run(&ArrayD::zeros(vec![]), &vec![&range]);
  let mut r = ArrayD::zeros(range.shape());
  let table_size = Fr::from(table_size as u32);
  azip!((&x in &range, &y in &result, z in &mut r) *z = x + y * table_size);
  r
}
