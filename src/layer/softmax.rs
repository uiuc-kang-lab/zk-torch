use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use std::collections::HashMap;

pub struct SoftmaxLayer;

impl Layer for SoftmaxLayer {
  fn load_onnx_layer(config: &LayerConfig) -> (Vec<usize>, Vec<Vec<(i32, usize)>>) {
    let blocks: Vec<String> = SoftmaxLayer::consume_basic_block(config).iter().map(|b| b.name()).collect();

    let blocks_idx = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];

    let nodes = vec![
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

    (blocks_idx, nodes)
  }

  fn consume_basic_block(config: &LayerConfig) -> Vec<Box<dyn BasicBlock>> {
    vec![
      Box::new(crate::basic_block::MaxBasicBlock),
      Box::new(crate::basic_block::SubBasicBlock),
      Box::new(crate::basic_block::ExpBasicBlock {
        input_SF: *config.input_params.get("input_SF").unwrap(),
        output_SF: *config.input_params.get("output_SF").unwrap(),
      }),
      Box::new(crate::basic_block::CQ2BasicBlock {
        table_dict: HashMap::new(),
        name: "Exp".to_string(),
      }),
      Box::new(crate::basic_block::UnsqueezeBasicBlock),
      Box::new(crate::basic_block::SumBasicBlock),
      Box::new(crate::basic_block::LogBasicBlock {
        input_SF: *config.input_params.get("input_SF").unwrap(),
        output_SF: *config.input_params.get("output_SF").unwrap(),
      }),
      Box::new(crate::basic_block::CQ2BasicBlock {
        table_dict: HashMap::new(),
        name: "Log".to_string(),
      }),
      Box::new(crate::basic_block::AddBasicBlock),
    ]
  }

  fn layer_output_node(config: &LayerConfig) -> (usize, usize) {
    (10, 0)
  }
}
