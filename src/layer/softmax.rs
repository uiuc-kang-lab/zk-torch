use super::{Layer, LayerConfig};
use crate::basic_block::BasicBlock;
use crate::graph::Node;
use std::collections::HashMap;

pub struct SoftmaxLayer;

impl Layer for SoftmaxLayer {
  fn load_onnx_layer(&self, config: &LayerConfig) -> Vec<Node> {
    let blocks: Vec<String> = self.consume_basic_block(config).iter().map(|b| b.name()).collect();

    let mut nodes = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];
    // if "shift_len" in config.input_params 
    if config.input_params.contains_key("shift_len") {
      let shift_len = config.input_params.get("shift_len").unwrap();
      nodes = nodes.iter().map(|x| x + shift_len).collect();
    }

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

  fn layer_output_node(&self, config: &LayerConfig) -> (usize, usize) {
    (10, 0)
  }
}
