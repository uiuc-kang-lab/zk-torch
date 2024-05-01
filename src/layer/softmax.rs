use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::{AddBasicBlock, CQ2BasicBlock, ExpBasicBlock, LogBasicBlock, MaxBasicBlock, SubBasicBlock, SumBasicBlock, UnsqueezeBasicBlock};
use std::collections::HashMap;

pub struct SoftmaxLayer;

impl Layer for SoftmaxLayer {
  fn load_layer_nodes(&self, config: &LayerConfig, basic_blocks: &Vec<Box<dyn BasicBlock>>) -> Vec<Node> {
    let basic_block_map: HashMap<&Box<dyn BasicBlock>, usize> = basic_blocks.iter().enumerate().map(|(i, b)| (b, i)).collect();
    let used_blocks: Vec<Box<dyn BasicBlock>> = self.consume_basic_block(config);

    let nodes: Vec<usize> = used_blocks.iter().map(|b| *basic_block_map.get(b).unwrap()).collect();

    // TODO: we need to handle axes from config later
    let inputs = vec![
      vec![(-1, 0)],         // max
      vec![(-1, 0), (0, 0)], // sub
      vec![(1, 0)],          // exp
      vec![(1, 0), (2, 0)],  // cq2
      vec![(2, 0)],          // unsqueeze
      vec![(4, 0)],          // sum
      vec![(5, 0)],          // log
      vec![(5, 0), (6, 0)],  // cq2
      vec![(0, 0), (6, 0)],  // add
      vec![(-1, 0), (8, 0)], // sub
      vec![(9, 0)],          // exp
      vec![(9, 0), (10, 0)], // cq2
    ];

    let layer = nodes.iter().zip(inputs).map(|(x, y)| Node { basic_block: *x, inputs: y }).collect();

    layer
  }

  fn consume_basic_block(&self, config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    let mut basic_blocks: Vec<Box<dyn BasicBlock>> = vec![
      Box::new(MaxBasicBlock),
      Box::new(SubBasicBlock),
      Box::new(ExpBasicBlock {
        input_SF: *config.input_params.get("input_SF").unwrap(),
        output_SF: *config.input_params.get("output_SF").unwrap(),
      }),
    ];
    basic_blocks.extend(vec![
      Box::new(CQ2BasicBlock {
        // TODO: make name have the input output sf
        table_dict: HashMap::new(),
        name: basic_blocks[basic_blocks.len() - 1].name(),
      }) as Box<dyn BasicBlock>,
      Box::new(UnsqueezeBasicBlock) as Box<dyn BasicBlock>,
      Box::new(SumBasicBlock) as Box<dyn BasicBlock>,
      Box::new(LogBasicBlock {
        input_SF: *config.input_params.get("input_SF").unwrap(),
        output_SF: *config.input_params.get("output_SF").unwrap(),
      }) as Box<dyn BasicBlock>,
    ]);
    basic_blocks.extend(vec![
      Box::new(CQ2BasicBlock {
        table_dict: HashMap::new(),
        name: basic_blocks[basic_blocks.len() - 1].name(),
      }) as Box<dyn BasicBlock>,
      Box::new(AddBasicBlock) as Box<dyn BasicBlock>,
    ]);
    basic_blocks
  }

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (8, 0)
  }
}
