/*
 * Verifier utilities:
 * The functions are used for verification-related operations, such as
 * an algorithm for combining pairing checks.
 */
use crate::basic_block::{BasicBlock, Data, DataEnc, PairingCheck, SRS};
use crate::graph::Graph;
use crate::util::msm;
use crate::{onnx, util, CONFIG, LAYER_SETUP_DIR};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::models::short_weierstrass::SWCurveConfig;
use ark_ec::pairing::PairingOutput;
use ark_ec::short_weierstrass::Affine;
use ark_ec::AffineRepr;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use ark_std::{One, Zero};
use ndarray::{arr0, arr1, concatenate, Array1, ArrayD, Axis, IxDyn};
use plonky2::{timed, util::timing::TimingTree};
use rand::{rngs::StdRng, SeedableRng};
use sha3::{Digest, Keccak256};
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::io::Read;

pub fn combine_pairing_checks(checks: &Vec<&PairingCheck>) -> (Vec<G1Affine>, Vec<G2Affine>) {
  println!("{:?}", checks.iter().map(|x| x.len()).sum::<usize>());

  let mut A = HashMap::new();
  let mut B = HashMap::new();
  let mut res: (Vec<G1Affine>, Vec<G2Affine>) = (Vec::new(), Vec::new());

  let mut rng = StdRng::from_entropy();
  let gamma = Fr::rand(&mut rng);
  #[cfg(feature = "fold")]
  let gamma = Fr::one(); // TODO: For folding, we use Fr::one() for now because there are errors to be summed up. This is a temporary fix.
  let mut curr = gamma;
  for check in checks.iter() {
    for pairing in check.iter() {
      if pairing.0.is_zero() || pairing.1.is_zero() {
        continue;
      }
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

pub fn verify(srs: &SRS, graph: &Graph, timing: &mut TimingTree) {
  // Read Files:
  let proofs =
    Vec::<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>::deserialize_uncompressed_unchecked(File::open(&CONFIG.verifier.proof_path).unwrap()).unwrap();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
  #[cfg(feature = "fold")]
  let acc_proofs = Vec::<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)>::deserialize_uncompressed_unchecked(
    File::open(&CONFIG.prover.acc_proof_path).unwrap(),
  )
  .unwrap();
  #[cfg(feature = "fold")]
  let acc_proofs = acc_proofs.iter().map(|x| (&x.0, &x.1, &x.2, &x.3)).collect();
  let mut modelsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_model_path).unwrap().read_to_end(&mut modelsEncBytes).unwrap();
  let modelsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&modelsEncBytes).unwrap();
  let modelsEnc: Vec<&ArrayD<DataEnc>> = modelsEnc.iter().map(|model| model).collect();
  let mut inputsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_input_path).unwrap().read_to_end(&mut inputsEncBytes).unwrap();
  let inputsEnc: Vec<ArrayD<DataEnc>> = bincode::deserialize(&inputsEncBytes).unwrap();
  let inputsEnc: Vec<&ArrayD<DataEnc>> = inputsEnc.iter().map(|input| input).collect();
  let mut outputsEncBytes = Vec::new();
  File::open(&CONFIG.verifier.enc_output_path).unwrap().read_to_end(&mut outputsEncBytes).unwrap();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> = bincode::deserialize(&outputsEncBytes).unwrap();
  let outputsEnc: Vec<Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputsEnc: Vec<&Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|x| x).collect();

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(modelsEncBytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);

  // Verify:
  #[cfg(all(feature = "debug", not(feature = "fold")))]
  timed!(
    timing,
    "verify (debug)",
    graph.verify_for_each_pairing(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng)
  );
  #[cfg(all(not(feature = "debug"), not(feature = "fold")))]
  timed!(
    timing,
    "verify",
    graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng, timing)
  );
  #[cfg(feature = "fold")]
  {
    let (final_proofs_idx, final_acc_proofs_idx) = timed!(
      timing,
      "verify",
      graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &acc_proofs, &mut rng, timing)
    );
    let final_proof = graph.fold_proofs(srs, final_proofs_idx, final_acc_proofs_idx, &proofs, &acc_proofs);
    final_proof.serialize_uncompressed(File::create(&CONFIG.prover.final_proof_path).unwrap()).unwrap();
  }
}
