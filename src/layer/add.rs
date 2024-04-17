use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use std::collections::HashMap;

pub struct AddLayer;

impl Layer for AddLayer {
  fn load_onnx_layer(&self, config: &LayerConfig) -> Vec<Node> {
    // if "shift_len" in config.input_params 
    let mut node = 0;
    if config.input_params.contains_key("shift_len") {
      let shift_len = config.input_params.get("shift_len").unwrap();
      node = *shift_len;
    }

    vec![
      Node { basic_block: node, inputs: vec![(-1, 0), (-1, 1)] }
    ]
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![Box::new(crate::basic_block::AddBasicBlock)]
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (0, 0)
  }
}
