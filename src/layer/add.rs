use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
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
}
