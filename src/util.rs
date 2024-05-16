#![allow(dead_code)]
use crate::{BasicBlock, Data, PairingCheck, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::models::short_weierstrass::SWCurveConfig;
use ark_ec::short_weierstrass::Affine;
use ark_ec::AffineRepr;
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, Zero};
use ndarray::{arr1, concatenate, Array1, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};
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

// For serialization, ArrayD uses serde while G1Affine uses ark_serialize.
// In order to bridge between the two, the following code snippet is used:
// https://github.com/arkworks-rs/algebra/issues/178#issuecomment-1413219278
pub fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  let mut bytes = vec![];
  a.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
  s.serialize_bytes(&bytes)
}

pub fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
  let a = A::deserialize_compressed_unchecked(s.as_slice());
  a.map_err(serde::de::Error::custom)
}

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
  let mut pow: Vec<Fr> = vec![alpha; n];
  for i in 0..n - 1 {
    pow[i + 1] = pow[i] * alpha;
  }
  pow
}

pub fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() == 0 {
    return arr1(&[Data::new(srs, a.view().as_slice().unwrap())]).into_dyn();
  }
  if a.ndim() == 1 {
    return arr1(&[Data::new(srs, a.as_slice().unwrap())]).into_dyn();
  }
  a.map_axis(Axis(a.ndim() - 1), |r| Data::new(srs, r.as_slice().unwrap()))
}

pub fn combine_pairing_checks(checks: &Vec<&PairingCheck>) -> (Vec<G1Affine>, Vec<G2Affine>) {
  println!("{:?}", checks.iter().map(|x| x.len()).sum::<usize>());

  let mut A = HashMap::new();
  let mut B = HashMap::new();
  let mut res: (Vec<G1Affine>, Vec<G2Affine>) = (Vec::new(), Vec::new());

  let mut rng = StdRng::from_entropy();
  let gamma = Fr::rand(&mut rng);
  let mut curr = gamma;
  for check in checks.iter() {
    for pairing in check.iter() {
      A.entry(pairing.0).or_insert_with(|| HashSet::new()).insert((pairing.1, curr));
      B.entry(pairing.1).or_insert_with(|| HashSet::new()).insert((pairing.0, curr));
    }
    curr *= gamma;
  }

  fn get_xy<P: SWCurveConfig>(a: &Affine<P>) -> (P::BaseField, P::BaseField) {
    let (x, y) = a.xy().unwrap();
    (*x, *y)
  }
  let mut ATree = BTreeSet::from_iter(A.iter().map(|(p, s)| (s.len(), get_xy(p))));
  let mut BTree = BTreeSet::from_iter(B.iter().map(|(p, s)| (s.len(), get_xy(p))));

  while !A.is_empty() {
    let (AAmt, _) = ATree.last().unwrap();
    let (BAmt, _) = BTree.last().unwrap();
    if AAmt > BAmt {
      // Combine G2 elements with the same G1 element
      let (_, AMax) = ATree.pop_last().unwrap();
      let AMax = G1Affine::new_unchecked(AMax.0, AMax.1);
      let (points, scalars): (Vec<G2Affine>, Vec<Fr>) = A.remove(&AMax).unwrap().into_iter().unzip();
      res.0.push(AMax);
      res.1.push(msm::<G2Projective>(&points, &scalars).into());
      for (p, r) in points.iter().zip(scalars) {
        let S = B.get_mut(&p).unwrap();
        let p2 = get_xy(p);
        BTree.remove(&(S.len(), p2));
        if S.len() == 1 {
          B.remove(&p);
        } else {
          S.remove(&(AMax, r));
          BTree.insert((S.len(), p2));
        }
      }
    } else {
      // Combine G1 elements with the same G2 element
      let (_, BMax) = BTree.pop_last().unwrap();
      let BMax: G2Affine = G2Affine::new_unchecked(BMax.0, BMax.1);
      let (points, scalars): (Vec<G1Affine>, Vec<Fr>) = B.remove(&BMax).unwrap().into_iter().unzip();
      res.0.push(msm::<G1Projective>(&points, &scalars).into());
      res.1.push(BMax);
      for (p, r) in points.iter().zip(scalars) {
        let S = A.get_mut(&p).unwrap();
        let p2 = get_xy(p);
        ATree.remove(&(S.len(), p2));
        if S.len() == 1 {
          A.remove(&p);
        } else {
          S.remove(&(BMax, r));
          ATree.insert((S.len(), p2));
        }
      }
    }
  }
  assert!(ATree.is_empty() && B.is_empty() && BTree.is_empty());
  println!("{:?}", res.0.len());
  res
}
