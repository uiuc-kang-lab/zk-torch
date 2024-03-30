use crate::basic_block::BasicBlock;
use crate::basic_block::*;
use crate::onnx_converter::{create_updated_models, load_onnx_tract_model, load_const_block, match_node_notes, match_node_op};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use rand::rngs::StdRng;
use std::error::Error;

pub struct Node {
  pub basic_block: usize,
  pub inputs: Vec<(i32, usize)>, //(node, output #)
}

pub struct Graph {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
}

impl Graph {
  pub fn build_from_onnx(path: &str) -> Result<(Self, Vec<Vec<Vec<Fr>>>), Box<dyn Error>> {
    let mut basic_blocks = vec![];
    let mut nodes = vec![];
    let mut models = vec![];
    let mut notes = match_node_notes::new(basic_blocks, nodes, models);

    let (tract_model, _) = load_onnx_tract_model(path).unwrap();

    let model_weight = load_const_block(&tract_model);

    match_node_op(tract_model, &mut notes);
    let updated_models = create_updated_models(&notes, 2); // small scale factor (i.e., 2) for now
    let basic_blocks = notes.basic_blocks;
    let nodes = notes.nodes;
    
    Ok((
      Self {
        basic_blocks: basic_blocks,
        nodes: nodes,
      },
      updated_models,
    ))
  }
  pub fn run(&self, inputs: &Vec<&Vec<Fr>>, models: &Vec<&Vec<&Vec<Fr>>>) -> Vec<Vec<Vec<Fr>>> {
    let mut outputs = vec![vec![]; self.nodes.len()];
    self.nodes.iter().enumerate().for_each(|(i, n)| {
      println!("running {i} {:?}", n.basic_block);
      let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
      outputs[i] = self.basic_blocks[n.basic_block].run(&models[n.basic_block], &myInputs);
    });
    return outputs;
  }
  pub fn setup(&self, srs: &SRS, models: &Vec<&Vec<&Data>>) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self.basic_blocks.iter().zip(models.iter()).map(|(b, m)| b.setup(srs, m)).collect()
  }
  pub fn prove(
    &mut self,
    srs: &SRS,
    setups: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    models: &Vec<&Vec<&Data>>,
    inputs: &Vec<&Data>,
    outputs: &Vec<&Vec<&Data>>,
    rng: &mut StdRng,
  ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs: Vec<_> = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        println!("proving {i} {}", myInputs.len());
        self.basic_blocks[n.basic_block].prove(srs, setups[n.basic_block], models[n.basic_block], &myInputs, outputs[i], rng)
      })
      .collect()
  }
  pub fn verify(
    &self,
    srs: &SRS,
    models: &Vec<&Vec<&DataEnc>>,
    inputs: &Vec<&DataEnc>,
    outputs: &Vec<&Vec<&DataEnc>>,
    proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
    rng: &mut StdRng,
  ) {
    self
      .nodes
      .iter()
      .enumerate()
      .map(|(i, n)| {
        let myInputs = n.inputs.iter().map(|(j, k)| if *j < 0 { inputs[*k] } else { &(outputs[*j as usize][*k]) }).collect();
        self.basic_blocks[n.basic_block].verify(srs, models[n.basic_block], &myInputs, outputs[i], proofs[i], rng)
      })
      .collect()
  }
}
