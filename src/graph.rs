#![allow(dead_code)]
use crate::basic_block::*;
use crate::util;
use crate::{CONFIG, LAYER_SETUP_DIR};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Precomputable {
  pub setup: Vec<bool>,
  pub prove_and_verify: Vec<bool>,
  pub encodeOutputs: Vec<bool>,
}

impl Precomputable {
  pub fn new() -> Self {
    Precomputable {
      setup: vec![],
      prove_and_verify: vec![],
      encodeOutputs: vec![],
    }
  }
}

#[derive(Debug)]
pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, // (node, output #)
}

#[derive(Debug)]
pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub precomputable: Precomputable,
  pub layer_names: Vec<String>,
  pub nodes: Vec<Node>,
  pub outputs: Vec<(i32, usize)>,
  #[cfg(feature = "fold")]
  pub foldable_bb_map: HashMap<usize, usize>,
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Result<Vec<Vec<ArrayD<Fr>>>, util::CQOutOfRangeError> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    let res: Result<(), util::CQOutOfRangeError> = self.nodes.iter().enumerate().try_for_each(|(i, n)| {
      println!("{} | running {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
      let myInputs = n
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if *basicblock_idx < 0 {
            // We currently support two types of indexing for the inputs, one is (-1,0),(-1,1),(-1,2),...
            // and the other is (-1,0),(-2,0),(-3,0),... In the future we will make this more standardized.
            inputs[*output_idx + (-basicblock_idx - 1) as usize]
          } else {
            &(outputs[*basicblock_idx as usize][*output_idx])
          }
        })
        .collect();
      outputs[i] = self.basic_blocks[n.basic_block].run(&models[n.basic_block], &myInputs)?;
      Ok(())
    });
    if res.is_err() {
      return Err(util::CQOutOfRangeError {
        input: res.err().unwrap().input,
      });
    }
    return Ok(outputs);
  }

  pub fn encodeOutputs(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Fr>>>,
    timing: &mut TimingTree,
  ) -> Vec<Vec<ArrayD<Data>>> {
    assert!(self.nodes.len() == self.precomputable.encodeOutputs.len());
    let mut outputsEnc = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      let precomputable = self.precomputable.encodeOutputs[i];
      if precomputable {
        // Skip encodeOutputs for some layers if they are precomputable.
        // These layers require no proving and verifying, and their outputs are not used as inputs of
        // `encodeOutputs` in any other layers that need proving and verifying.
        println!(
          "{} | skipping encodingOutputs for {i} {:?} because the output is precomputable and will not be used as input in any layer that needs proving and verifying",
          self.layer_names[i], self.basic_blocks[n.basic_block]
        );
        return;
      }
      let encode_id = format!("{} | encoding node {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
      println!("{}", encode_id);
      let myInputs = n
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if *basicblock_idx < 0 {
            inputs[*output_idx + (-basicblock_idx - 1) as usize]
          } else {
            &(outputsEnc[*basicblock_idx as usize][*output_idx])
          }
        })
        .collect();
      outputsEnc[i] = timed!(
        timing,
        &encode_id,
        self.basic_blocks[n.basic_block].encodeOutputs(srs, &models[n.basic_block], &myInputs, outputs[i])
      );
    });
    return outputsEnc;
  }

  pub fn setup(&self, srs: &SRS, models: &Vec<&ArrayD<Data>>) -> Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)> {
    assert!(self.basic_blocks.len() == self.precomputable.setup.len());
    self
      .basic_blocks
      .iter()
      .zip(models.iter())
      .enumerate()
      .map(|(i, (b, m))| {
        let precomputable = self.precomputable.setup[i];
        if precomputable {
          // Skip setup for some basicblocks if they are precomputable.
          // These basicblocks require no proving and verifying since they are not used in any layer that needs proving and verifying.
          println!(
            "skipping setup for {:?} {:?} because the basicblock is not used in any layer that needs proving and verifying",
            i, b
          );
          return (vec![], vec![], vec![]);
        }
        println!("setting up {:?} {:?}", i, b);
        let bb_name = format!("{b:?}");
        let save_cq_layer_setup = CONFIG.prover.enable_layer_setup && (bb_name.contains("CQ2BasicBlock") || bb_name.contains("CQBasicBlock"));
        #[cfg(not(feature = "mock_prove"))]
        if save_cq_layer_setup {
          let file_name = format!("{}.setup", util::hash_str(&format!("{bb_name:?}")));
          let file_path = format!("{}/{}", *LAYER_SETUP_DIR, file_name);
          if util::file_exists(&file_path) {
            println!("CQ setup exists: Loading layer setup from file: {}", file_path);
            let setups =
              Vec::<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>::deserialize_uncompressed(File::open(&file_path).unwrap())
                .unwrap();
            return setups.first().unwrap().clone();
          }
        }
        let setup = b.setup(srs, *m);
        let setups = vec![setup];
        #[cfg(not(feature = "mock_prove"))]
        if save_cq_layer_setup {
          let file_name = format!("{}.setup", util::hash_str(&format!("{bb_name:?}")));
          let file_path = format!("{}/{}", *LAYER_SETUP_DIR, file_name);
          setups.serialize_uncompressed(File::create(file_path).unwrap()).unwrap();
        }
        return setups.first().unwrap().clone();
      })
      .collect()
  }

  #[cfg(not(feature = "fold"))]
  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)>,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    rng: &mut StdRng,
    timing: &mut TimingTree,
  ) -> Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)> {
    assert!(self.nodes.len() == self.precomputable.prove_and_verify.len());
    let cache = Arc::new(Mutex::new(HashMap::new()));

    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let precomputable = self.precomputable.prove_and_verify[i];
        if precomputable {
          // Skip proving for some layers if they are precomputable.
          // These layers require no proving and verifying as their inputs are known (i.e., constants) during graph construction.
          println!(
            "{} | skipping proving for {i} {:?} because this layer is precomputable given the constant inputs",
            self.layer_names[i], self.basic_blocks[n.basic_block]
          );
          return (vec![], vec![], vec![]);
        }
        let prove_id = format!("{} | proving {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
        println!("{}", prove_id);
        let myInputs = n
          .inputs
          .iter()
          .map(|(basicblock_idx, output_idx)| {
            if *basicblock_idx < 0 {
              inputs[*output_idx + (-basicblock_idx - 1) as usize]
            } else {
              &(outputs[*basicblock_idx as usize][*output_idx])
            }
          })
          .collect();
        let proof = timed!(
          timing,
          &prove_id,
          self.basic_blocks[n.basic_block].prove(
            srs,
            setups[n.basic_block],
            models[n.basic_block],
            &myInputs,
            outputs[i],
            rng,
            cache.clone(),
          )
        );
        let proof: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (
          proof.0.iter().map(|x| (*x).into()).collect(),
          proof.1.iter().map(|x| (*x).into()).collect(),
          proof.2.iter().map(|x| (*x).into()).collect(),
        );
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        util::add_randomness(rng, bytes);
        proof
      })
      .collect()
  }

  #[cfg(feature = "fold")]
  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)>,
    models: &Vec<&ArrayD<Data>>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&Vec<&ArrayD<Data>>>,
    rng: &mut StdRng,
    timing: &mut TimingTree,
  ) -> (
    Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>,
    Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)>,
  ) {
    assert!(self.nodes.len() == self.precomputable.prove_and_verify.len());
    let cache = Arc::new(Mutex::new(HashMap::new()));
    let mut proofs = vec![];
    let mut acc_proofs_for_verifier = vec![];
    let mut prev_acc_map: HashMap<usize, (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)> = HashMap::new(); // (basicblock, acc_proof)

    self.nodes.iter().enumerate().for_each(|(i, n)| {
      let bb_index_for_folding = self.foldable_bb_map.get(&n.basic_block).unwrap();
      let precomputable = self.precomputable.prove_and_verify[i];
      if precomputable {
        // Skip proving for some layers if they are precomputable.
        // These layers require no proving and verifying as their inputs are known (i.e., constants) during graph construction.
        println!(
          "{} | skipping proving for {i} {:?} because this layer is precomputable given the constant inputs",
          self.layer_names[i], self.basic_blocks[n.basic_block]
        );
        proofs.push((vec![], vec![], vec![]));
        prev_acc_map.insert(*bb_index_for_folding, (vec![], vec![], vec![], vec![]));
        acc_proofs_for_verifier.push((vec![], vec![], vec![], vec![]));
        return;
      }
      let prove_id = format!("{} | proving {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
      println!("{}", prove_id);
      let myInputs = n
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if *basicblock_idx < 0 {
            inputs[*output_idx + (-basicblock_idx - 1) as usize]
          } else {
            &(outputs[*basicblock_idx as usize][*output_idx])
          }
        })
        .collect();
      let proof = timed!(
        timing,
        &prove_id,
        self.basic_blocks[n.basic_block].prove(
          srs,
          setups[n.basic_block],
          models[n.basic_block],
          &myInputs,
          outputs[i],
          rng,
          cache.clone(),
        )
      );

      let mut prev_acc_proof: (
        &Vec<G1Projective>,
        &Vec<G2Projective>,
        &Vec<Fr>,
        &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
      ) = (&vec![], &vec![], &vec![], &vec![]);

      // check if basicblock is in prev_acc_map
      if let Some(prev_acc) = prev_acc_map.get(bb_index_for_folding) {
        prev_acc_proof = (&prev_acc.0, &prev_acc.1, &prev_acc.2, &prev_acc.3);
      }

      let new_acc_proof = self.basic_blocks[n.basic_block].acc_prove(
        srs,
        models[n.basic_block],
        &myInputs,
        outputs[i],
        prev_acc_proof,
        (&proof.0, &proof.1, &proof.2),
        rng,
        cache.clone(),
      );

      let (proof, new_acc_proof_v) = self.basic_blocks[self.nodes[i].basic_block].acc_clean(
        srs,
        (&proof.0, &proof.1, &proof.2),
        (&new_acc_proof.0, &new_acc_proof.1, &new_acc_proof.2, &new_acc_proof.3),
      );

      proofs.push(proof);
      acc_proofs_for_verifier.push(new_acc_proof_v);
      prev_acc_map.insert(*bb_index_for_folding, new_acc_proof);

      let mut bytes = Vec::new();
      proofs[proofs.len() - 1].serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
    });

    (proofs, acc_proofs_for_verifier)
  }

  #[cfg(not(feature = "fold"))]
  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)>,
    rng: &mut StdRng,
    timing: &mut TimingTree,
  ) {
    assert!(self.nodes.len() == self.precomputable.prove_and_verify.len());
    let cache = Arc::new(Mutex::new(HashMap::new()));

    let pairings: Vec<Vec<PairingCheck>> = self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let precomputable = self.precomputable.prove_and_verify[i];
        if precomputable {
          // Skip verifying for some layers if they are precomputable.
          // These layers require no proving and verifying as their inputs are known (i.e., constants) during graph construction.
          println!(
            "{} | skipping verifying for {i} {:?} because this layer is precomputable given the constant inputs",
            self.layer_names[i], self.basic_blocks[n.basic_block]
          );
          return vec![];
        }
        let verify_id = format!("{} | verifying {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
        println!("{}", verify_id);
        let myInputs = n
          .inputs
          .iter()
          .map(|(basicblock_idx, output_idx)| {
            if *basicblock_idx < 0 {
              inputs[*output_idx + (-basicblock_idx - 1) as usize]
            } else {
              &(outputs[*basicblock_idx as usize][*output_idx])
            }
          })
          .collect();
        let pairings = timed!(
          timing,
          &verify_id,
          self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, cache.clone())
        );
        let mut bytes = Vec::new();
        let temp: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (proofs[i].0.clone(), proofs[i].1.clone(), proofs[i].2.clone());
        temp.serialize_uncompressed(&mut bytes).unwrap();
        util::add_randomness(rng, bytes);
        pairings
      })
      .collect();
    let pairings = timed!(
      timing,
      "combine pairings",
      util::combine_pairing_checks(&pairings.iter().flatten().collect())
    );
    let pairing_check = timed!(timing, "pairings", Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()));
    assert_eq!(pairing_check, PairingOutput::zero());
  }

  #[cfg(feature = "fold")]
  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)>,
    acc_proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>)>,
    rng: &mut StdRng,
    timing: &mut TimingTree,
  ) -> (Vec<usize>, Vec<usize>) {
    assert!(self.nodes.len() == self.precomputable.prove_and_verify.len());
    let cache = Arc::new(Mutex::new(HashMap::new()));
    let mut prev_acc_map: HashMap<usize, usize> = HashMap::new();
    let (mut final_proofs_idx, mut final_acc_proofs_idx) = (vec![], vec![]);

    let mut pairings: Vec<Vec<PairingCheck>> = self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let precomputable = self.precomputable.prove_and_verify[i];
        if precomputable {
          // Skip verifying for some layers if they are precomputable.
          // These layers require no proving and verifying as their inputs are known (i.e., constants) during graph construction.
          println!(
            "{} | skipping verifying for {i} {:?} because this layer is precomputable given the constant inputs",
            self.layer_names[i], self.basic_blocks[n.basic_block]
          );
          return vec![];
        }
        let verify_id = format!("{} | verifying {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
        let acc_verify_id = format!("{} | acc verifying {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
        println!("{}", verify_id);
        let myInputs = n
          .inputs
          .iter()
          .map(|(basicblock_idx, output_idx)| {
            if *basicblock_idx < 0 {
              inputs[*output_idx + (-basicblock_idx - 1) as usize]
            } else {
              &(outputs[*basicblock_idx as usize][*output_idx])
            }
          })
          .collect();

        let bb_index_for_folding = self.foldable_bb_map.get(&n.basic_block).unwrap();
        let mut prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>) =
          (&vec![], &vec![], &vec![], &vec![]);
        if let Some(prev_acc) = prev_acc_map.get(bb_index_for_folding) {
          prev_acc_proof = (
            &acc_proofs[*prev_acc].0,
            &acc_proofs[*prev_acc].1,
            &acc_proofs[*prev_acc].2,
            &acc_proofs[*prev_acc].3,
          );
        }
        prev_acc_map.insert(*bb_index_for_folding, i);

        let acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>) =
          (&acc_proofs[i].0, &acc_proofs[i].1, &acc_proofs[i].2, &acc_proofs[i].3);
        let pairings = timed!(
          timing,
          &verify_id,
          self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, cache.clone())
        );

        let acc_verification = timed!(
          timing,
          &acc_verify_id,
          self.basic_blocks[n.basic_block].acc_verify(
            srs,
            models[n.basic_block],
            &myInputs,
            outputs[i],
            prev_acc_proof,
            acc_proof,
            (&proofs[i].0, &proofs[i].1, &proofs[i].2),
            rng,
            cache.clone()
          )
        );

        if acc_verification.is_none() {
          final_proofs_idx.push(i);
        } else {
          #[cfg(not(feature = "mock_prove"))]
          assert!(acc_verification.unwrap(), "Accumulator verification failed");
        };
        let mut bytes = Vec::new();
        let temp: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (proofs[i].0.clone(), proofs[i].1.clone(), proofs[i].2.clone());
        temp.serialize_uncompressed(&mut bytes).unwrap();
        util::add_randomness(rng, bytes);
        pairings
      })
      .collect();

    let mut err_collector = vec![];
    let mut decider_pairings: Vec<Vec<PairingCheck>> = prev_acc_map
      .iter()
      .map(|(k, v)| {
        if !final_proofs_idx.contains(v) {
          final_acc_proofs_idx.push(*v);
        }
        let acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>) =
          (&acc_proofs[*v].0, &acc_proofs[*v].1, &acc_proofs[*v].2, &acc_proofs[*v].3);
        let decider_result = self.basic_blocks[*k].acc_decide(srs, acc_proof);
        err_collector.push(decider_result.iter().map(|x| x.1).collect::<Vec<PairingOutput<Bn<ark_bn254::Config>>>>());
        decider_result.iter().map(|x| x.0.clone()).collect::<Vec<PairingCheck>>()
      })
      .collect();
    let err_sum = err_collector.iter().flatten().fold(
      PairingOutput::<Bn254>::zero(), 
      |acc, x| acc + x
    );
    pairings.append(&mut decider_pairings);

    let pairings = timed!(
      timing,
      "combine pairings",
      util::combine_pairing_checks(&pairings.iter().flatten().collect())
    );
    let pairing_check = timed!(timing, "pairings", Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()));
    //assert_eq!(pairing_check, PairingOutput::zero());
    println!("Is verification successful? {}", pairing_check == err_sum);
    (final_proofs_idx, final_acc_proofs_idx)
  }

  #[cfg(feature = "fold")]
  pub fn fold_proofs(
    &self,
    srs: &SRS,
    final_proofs_idx: Vec<usize>,
    final_acc_proofs_idx: Vec<usize>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)>,
    acc_proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>)>,
  ) -> Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)> {
    let mut final_proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)> =
      final_proofs_idx.iter().map(|i| (proofs[*i].0.clone(), proofs[*i].1.clone(), proofs[*i].2.clone(), vec![])).collect();
    let mut final_acc_proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)> = final_acc_proofs_idx
      .iter()
      .map(|i| {
        let n = &self.nodes[*i];
        self.basic_blocks[n.basic_block].acc_finalize(srs, (&acc_proofs[*i].0, &acc_proofs[*i].1, &acc_proofs[*i].2, &acc_proofs[*i].3))
      })
      .collect();
    final_proofs.append(&mut final_acc_proofs);
    final_proofs
  }

  // This function should be only used for debugging purposes (it is very slow).
  // It verifies the proofs without combining pairing checks so that we can see which BasicBlock is failing.
  pub fn verify_for_each_pairing(
    &self,
    srs: &SRS,
    models: &Vec<&ArrayD<DataEnc>>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&Vec<&ArrayD<DataEnc>>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)>,
    rng: &mut StdRng,
  ) {
    assert!(self.nodes.len() == self.precomputable.prove_and_verify.len());
    let cache = Arc::new(Mutex::new(HashMap::new()));

    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("verifying (debug mode) {i} {:?}", self.basic_blocks[n.basic_block]);
      let precomputable = self.precomputable.prove_and_verify[i];
      if precomputable {
        // Skip verifying for some layers if they are precomputable.
        // These layers require no proving and verifying as their inputs are known (i.e., constants) during graph construction.
        println!(
          "{} | skipping verifying for {i} {:?} because this layer is precomputable given the constant inputs",
          self.layer_names[i], self.basic_blocks[n.basic_block]
        );
        return;
      }
      let myInputs = n
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if *basicblock_idx < 0 {
            inputs[*output_idx + (-basicblock_idx - 1) as usize]
          } else {
            &(outputs[*basicblock_idx as usize][*output_idx])
          }
        })
        .collect();
      if !precomputable {
        let pairings = self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, cache.clone());
        let mut bytes = Vec::new();
        let temp: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (proofs[i].0.clone(), proofs[i].1.clone(), proofs[i].2.clone());
        temp.serialize_uncompressed(&mut bytes).unwrap();
        util::add_randomness(rng, bytes);
        pairings.iter().for_each(|p| {
          assert!(p
            .iter()
            .fold(PairingOutput::<Bn<ark_bn254::Config>>::zero(), |acc, x| {
              acc + Bn254::pairing(x.0, x.1)
            })
            .is_zero());
        });
      }
    });
  }

  pub fn new() -> Self {
    Graph {
      basic_blocks: vec![],
      precomputable: Precomputable::new(),
      layer_names: vec![],
      nodes: vec![],
      outputs: vec![],
      #[cfg(feature = "fold")]
      foldable_bb_map: HashMap::new(),
    }
  }

  pub fn addBB(&mut self, basic_block: Box<dyn BasicBlock>) -> usize {
    self.basic_blocks.push(basic_block);
    self.basic_blocks.len() - 1
  }

  pub fn addNode(&mut self, basic_block: usize, inputs: Vec<(i32, usize)>) -> i32 {
    self.nodes.push(Node {
      basic_block: basic_block,
      inputs: inputs,
    });
    self.layer_names.push("Precomputation".to_string());
    (self.nodes.len() - 1) as i32
  }
}
