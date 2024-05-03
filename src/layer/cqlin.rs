use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::{CQLinBasicBlock, SqueezeBasicBlock};
use std::collections::HashMap;

// Only used for testing purposes
pub struct CQLinLayer;

impl Layer for CQLinLayer {
  fn layer_inputs(&self) -> Vec<Vec<(i32, usize)>> {
    vec![vec![(-1, 0)], vec![(0, 0)]]
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![
      Box::new(CQLinBasicBlock {
        weights_name: config.weights_names[0].clone(),
      }),
      Box::new(SqueezeBasicBlock),
    ]
  }

  fn layer_output_node(&self, _config: &LayerConfig) -> (usize, usize) {
    (1, 0)
  }
}
