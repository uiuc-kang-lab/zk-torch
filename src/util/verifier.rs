/*
 * Verifier utilities:
 * The functions are used for verification-related operations, such as
 * an algorithm for combining pairing checks.
 */
use crate::util::msm;
use crate::PairingCheck;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::models::short_weierstrass::SWCurveConfig;
use ark_ec::short_weierstrass::Affine;
use ark_ec::AffineRepr;
use ark_std::UniformRand;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};

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
