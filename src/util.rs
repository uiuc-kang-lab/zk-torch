#![allow(dead_code)]
use crate::{BasicBlock, Data, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::models::short_weierstrass::SWCurveConfig;
use ark_ec::short_weierstrass::Affine;
use ark_ec::AffineRepr;
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{UniformRand, Zero};
use ndarray::{arr0, concatenate, Array1, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};

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

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, offset: i32, size: usize) -> ArrayD<Fr> {
  let range = Array1::from_shape_fn(size, |i| Fr::from(i as u32) + Fr::from(offset)).into_dyn();
  let result = &(**basic_block).run(&ArrayD::zeros(IxDyn(&[0])), &vec![&range])[0];
  let range = range.view().into_shape(IxDyn(&[1, size])).unwrap();
  let result = result.view().into_shape(IxDyn(&[1, size])).unwrap();
  concatenate(Axis(0), &[range, result]).unwrap()
}

pub fn fr_to_int(x: Fr) -> i32 {
  if x < Fr::from(1 << 28) {
    x.into_bigint().0[0] as i32
  } else {
    -((-x).into_bigint().0[0] as i32)
  }
}

pub fn calc_pow(alpha: Fr, n: usize) -> Vec<Fr> {
  // Starts at alpha^1 for AlternatingBasicBlock to distinguish first elements
  let mut pow: Vec<Fr> = vec![alpha; n];
  for i in 0..n - 1 {
    pow[i + 1] = pow[i] * alpha;
  }
  pow
}

pub fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() == 0 {
    return arr0(Data::new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  a.map_axis(Axis(a.ndim() - 1), |r| Data::new(srs, r.as_slice().unwrap()))
}

pub fn combine_pairing_checks(checks: &Vec<&Vec<(G1Affine, G2Affine)>>) -> (Vec<G1Affine>, Vec<G2Affine>) {
  println!("{:?}", checks.iter().map(|x| x.len()).sum::<usize>());

  let mut a = HashMap::new();
  let mut b = HashMap::new();
  let mut res: (Vec<G1Affine>, Vec<G2Affine>) = (Vec::new(), Vec::new());

  let mut rng = StdRng::from_entropy();
  let gamma = Fr::rand(&mut rng);
  let mut curr = gamma;
  for eqn in checks.iter() {
    for pairing in eqn.iter() {
      a.entry(pairing.0).or_insert(HashSet::new()).insert((pairing.1, curr));
      b.entry(pairing.1).or_insert(HashSet::new()).insert((pairing.0, curr));
    }
    curr *= gamma;
  }

  fn get_xy<P: SWCurveConfig>(a: &Affine<P>) -> (P::BaseField, P::BaseField) {
    let (x, y) = a.xy().unwrap();
    (*x, *y)
  }
  let mut a2 = BTreeSet::from_iter(a.iter().map(|(x, y)| (y.len(), get_xy(x))));
  let mut b2 = BTreeSet::from_iter(b.iter().map(|(x, y)| (y.len(), get_xy(x))));

  while b.len() > 0 {
    let (ax, _) = a2.last().unwrap();
    let (bx, _) = b2.last().unwrap();
    if ax > bx {
      // Greedily combine g1 elements
      let (_, ay) = a2.pop_last().unwrap();
      let temp: G1Affine = G1Affine::new_unchecked(ay.0, ay.1);
      res.0.push(temp);
      res.1.push(a[&temp].iter().map(|(x, y)| *x * y).sum::<G2Projective>().into());
      for (x, y) in a[&temp].iter() {
        let y1 = get_xy(x);
        b2.remove(&(b[&x].len(), y1));
        let temp2 = b.get_mut(&x).unwrap();
        if temp2.len() == 1 {
          b.remove(&x);
        } else {
          temp2.remove(&(temp, *y));
          b2.insert((b[&x].len(), y1));
        }
      }
      a.remove(&temp);
    } else {
      // Greedily combine g2 elements
      let (_, ay) = b2.pop_last().unwrap();
      let temp: G2Affine = G2Affine::new_unchecked(ay.0, ay.1);
      res.0.push(b[&temp].iter().map(|(x, y)| *x * y).sum::<G1Projective>().into());
      res.1.push(temp);
      for (x, y) in b[&temp].iter() {
        let y1 = get_xy(x);
        a2.remove(&(a[&x].len(), y1));
        let temp2 = a.get_mut(&x).unwrap();
        if temp2.len() == 1 {
          a.remove(&x);
        } else {
          temp2.remove(&(temp, *y));
          a2.insert((a[&x].len(), y1));
        }
      }
      b.remove(&temp);
    }
  }
  println!("{:?}", res.0.len());
  res
}
