use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::{CQLinBasicBlock, SqueezeBasicBlock};
use std::collections::HashMap;

pub struct CQLinLayer;

impl Layer for CQLinLayer {
  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
    let basic_block_map: HashMap<&Box<dyn BasicBlock>, usize> = basic_blocks.iter().enumerate().map(|(i, b)| (b, i)).collect();
    let used_blocks: Vec<Box<dyn BasicBlock>> = self.consume_basic_block(config);

    let nodes: Vec<usize> = used_blocks.iter().map(|b| *basic_block_map.get(b).unwrap()).collect();

    let inputs = vec![vec![(-1, 0)], vec![(0, 0)]];

    let layer = nodes.iter().zip(inputs).map(|(x, y)| Node { basic_block: *x, inputs: y }).collect();

    layer
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
