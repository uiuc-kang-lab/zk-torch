use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use std::collections::HashMap;

pub struct AddLayer;

impl Layer for AddLayer {
  fn load_onnx_layer(config: &LayerConfig) -> (Vec<usize>, Vec<Vec<(i32, usize)>>) {
    (vec![0], vec![vec![(-1, 0), (-1, 1)]])
  }

  fn consume_basic_block(config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![Box::new(crate::basic_block::AddBasicBlock)]
  }

  fn layer_output_node(config: &LayerConfig) -> (usize, usize) {
    (0, 0)
  }
}
