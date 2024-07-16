use crate::basic_block::*;
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::CanonicalSerialize;
use ark_std::Zero;
use ndarray::ArrayD;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, // (node, output #)
}

#[derive(Debug)]
pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
  pub outputs: Vec<(i32, usize)>,
}

impl Graph {
  pub fn run(&self, inputs: &Vec<&ArrayD<Fr>>, models: &Vec<&ArrayD<Fr>>) -> Vec<Vec<ArrayD<Fr>>> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", self.basic_blocks[n.basic_block]);
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
  ) -> Vec<Vec<ArrayD<Data>>> {
    let mut outputsEnc = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("encoding {i} {:?}", self.basic_blocks[n.basic_block]);
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
      outputsEnc[i] = self.basic_blocks[n.basic_block].encodeOutputs(srs, &models[n.basic_block], &myInputs, outputs[i]);
    });
    return outputsEnc;
  }

  pub fn setup(&self, srs: &SRS, models: &Vec<&ArrayD<Data>>) -> Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)> {
    self
      .basic_blocks
      .iter()
      .zip(models.iter())
      .enumerate()
      .map(|(i, (b, m))| {
        println!("setting up {:?} {:?}", i, b);
        b.setup(srs, *m)
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
  ) -> Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)> {
    let mut cache = HashMap::new();
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        println!("proving {i} {:?}", self.basic_blocks[n.basic_block]);
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
        let proof = self.basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &myInputs, outputs[i], rng, &mut cache);
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
    let mut cache = HashMap::new();
    let pairings: Vec<Vec<PairingCheck>> = self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        println!("verifying {i} {:?}", self.basic_blocks[n.basic_block]);
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
        let pairings = self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, &mut cache);
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
    let mut cache = HashMap::new();
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("verifying (debug mode) {i} {:?}", self.basic_blocks[n.basic_block]);
      let myInputs = n
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if *basicblock_idx < 0 {
            inputs[*output_idx]
          } else {
            &(outputs[*basicblock_idx as usize][*output_idx])
          }
        })
        .collect();
      let pairings = self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng, &mut cache);
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
    });
  }

  pub fn new() -> Self {
    Graph {
      basic_blocks: vec![],
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
    (self.nodes.len() - 1) as i32
  }
}
