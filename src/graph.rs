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
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Vec<Vec<ArrayD<Fr>>> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
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
      outputs[i] = self.basic_blocks[n.basic_block].run(&models[n.basic_block], &myInputs);
    });
    return outputs;
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
        if save_cq_layer_setup {
          let file_name = format!("{}.setup", util::hash_str(&format!("{bb_name:?}")));
          let file_path = format!("{}/{}", *LAYER_SETUP_DIR, file_name);
          setups.serialize_uncompressed(File::create(file_path).unwrap()).unwrap();
        }
        return setups.first().unwrap().clone();
      })
      .collect()
  }

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

  pub fn verify(
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
        println!("{} | verifying {i} {:?}", self.layer_names[i], self.basic_blocks[n.basic_block]);
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
        let pairings = self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, cache.clone());
        let mut bytes = Vec::new();
        let temp: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (proofs[i].0.clone(), proofs[i].1.clone(), proofs[i].2.clone());
        temp.serialize_uncompressed(&mut bytes).unwrap();
        util::add_randomness(rng, bytes);
        pairings
      })
      .collect();
    let pairings = util::combine_pairing_checks(&pairings.iter().flatten().collect());
    assert_eq!(Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()), PairingOutput::zero());
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
      let precomputable = self.precomputable.prove_and_verify[i];
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
