use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::{Node, Setup, SetupType};
use crate::util::convert_to_data;
use crate::DataEnc;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{ArrayD, Axis, IxDyn, Zip};
use rand::rngs::StdRng;
use std::collections::HashMap;

pub struct AddLayer;

impl Layer for AddLayer {
  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
    let basic_block_map: HashMap<&Box<dyn BasicBlock>, usize> = basic_blocks.iter().enumerate().map(|(i, b)| (b, i)).collect();
    let used_blocks: Vec<Box<dyn BasicBlock>> = self.consume_basic_block(config);

    let node = *basic_block_map.get(&used_blocks[0]).unwrap();

    vec![Node {
      basic_block: node,
      inputs: vec![(-1, 0), (-1, 1)],
    }]
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![Box::new(crate::basic_block::AddBasicBlock)]
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (0, 0)
  }

  fn run(
    &self,
    nodes: &Vec<Node>,
    inputs: &Vec<&ArrayD<Fr>>,
    weights: &Vec<&ArrayD<Fr>>,
    basic_blocks: &Vec<Box<dyn BasicBlock>>,
  ) -> Vec<Vec<ArrayD<Fr>>> {
    let n = &nodes[0]; // only one node: Add
    println!("running {:?}", n.basic_block);
    let inputs: Vec<_> = n.inputs.iter().map(|(_j, k)| inputs[*k]).collect();
    let a = inputs[0];
    let b = inputs[1];
    if a.ndim() > b.ndim() {
      let b = &b.broadcast(a.raw_dim()).unwrap().to_owned();
    } else if a.ndim() < b.ndim() {
      let a = &a.broadcast(b.raw_dim()).unwrap().to_owned();
    }
    let broadcast_shape = a.raw_dim();
    let ndim = a.ndim();
    let mut c = ArrayD::<Fr>::zeros(broadcast_shape);

    Zip::from(c.lanes_mut(Axis(ndim - 1))).and(a.lanes(Axis(ndim - 1))).and(b.lanes(Axis(ndim - 1))).for_each(|mut c, a, b| {
      let c_owned = basic_blocks[n.basic_block].run(&weights[n.basic_block], &vec![&a.to_owned().into_dyn(), &b.to_owned().into_dyn()])[0].clone();
      c.assign(&c_owned);
    });

    vec![vec![c]]
  }

  fn prove(
      &self,
      nodes: &mut &Vec<Node>,
      srs: &crate::SRS,
      setups: &Setup,
      inputs: &Vec<&ArrayD<crate::Data>>,
      outputs: &Vec<&Vec<&ArrayD<crate::Data>>>,
      basic_blocks: &mut Vec<Box<dyn BasicBlock>>,
      rng: &mut StdRng,
    ) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
    let mut results = vec![];
    let n = &nodes[0]; // only one node: Add
    let inputs: Vec<_> = n.inputs.iter().map(|(_j, k)| inputs[*k]).collect();
    let a = inputs[0];
    let b = inputs[1];
    if a.ndim() > b.ndim() {
      let b = &b.broadcast(a.raw_dim()).unwrap();
    } else if a.ndim() < b.ndim() {
      let a = &a.broadcast(b.raw_dim()).unwrap();
    }
    let broadcast_shape = a.raw_dim();
    let ndim = a.ndim();
    if ndim > 0 {
      let a_lanes = a.lanes(Axis(ndim - 1));
      let b_lanes = b.lanes(Axis(ndim - 1));
      let c_lanes = outputs[0][0].lanes(Axis(ndim - 1));
      
      a_lanes.into_iter().zip(b_lanes.into_iter()).zip(c_lanes.into_iter()).for_each(|((a, b), c)| {
      let a = a.to_owned().into_dyn();
      let b = b.to_owned().into_dyn();
      let c = c.to_owned().into_dyn();
      let (g1, g2) = basic_blocks[n.basic_block].prove(srs, &SetupType::None, &vec![&a, &b], &vec![&c], rng);
      results.push((g1, g2));
      });
      results
    } else {
      let c = outputs[0][0];
      let (g1, g2) = basic_blocks[n.basic_block].prove(srs, &SetupType::None, &vec![&a, &b], &vec![&c], rng);
      results.push((g1, g2));
      results
    }
  }

  fn verify(
      &self,
      nodes: &Vec<Node>,
      srs: &crate::SRS,
      weights: &HashMap<String, ArrayD<crate::DataEnc>>,
      tables: &HashMap<String, ArrayD<crate::DataEnc>>,
      inputs: &Vec<&ArrayD<crate::DataEnc>>,
      outputs: &Vec<&Vec<&ArrayD<crate::DataEnc>>>,
      proofs: &Vec<(&Vec<G1Affine>, &Vec<G2Affine>)>,
      basic_blocks: &Vec<Box<dyn BasicBlock>>,
      rng: &mut StdRng,
    ) {
    let n = &nodes[0]; // only one node: Add
    let inputs: Vec<_> = n.inputs.iter().map(|(_j, k)| inputs[*k]).collect();
    let a = inputs[0];
    let b = inputs[1];
    if a.ndim() > b.ndim() {
      let b = &b.broadcast(a.raw_dim()).unwrap();
    } else if a.ndim() < b.ndim() {
      let a = &a.broadcast(b.raw_dim()).unwrap();
    }
    let broadcast_shape = a.raw_dim();
    let ndim = a.ndim();
    let empty = convert_to_data(&srs, &ArrayD::zeros(IxDyn(&[0]))).map(|x| DataEnc::new(srs, x));
    if ndim > 0 {
      let a_lanes = a.lanes(Axis(ndim - 1));
      let b_lanes = b.lanes(Axis(ndim - 1));
      let c_lanes = outputs[0][0].lanes(Axis(ndim - 1));
      let mut proof_counter = 0;

      
      a_lanes.into_iter().zip(b_lanes.into_iter()).zip(c_lanes.into_iter()).for_each(|((a, b), c)| {
        let a = a.to_owned().into_dyn();
        let b = b.to_owned().into_dyn();
        let c = c.to_owned().into_dyn();
        basic_blocks[n.basic_block].verify(srs, &empty, &vec![&a, &b], &vec![&c], proofs[proof_counter], rng);
        proof_counter += 1;
      });
    } else {
      let c = outputs[0][0];
      basic_blocks[n.basic_block].verify(srs, &empty, &vec![&a, &b], &vec![&c], proofs[0], rng);
    }
  }
}
