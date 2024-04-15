#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::basic_block::BasicBlock;
use crate::graph::Node;
pub use add::AddLayer;
pub use softmax::SoftmaxLayer;
use std::collections::HashMap;

pub mod add;
pub mod softmax;

pub struct LayerConfig {
  pub input_params: HashMap<String, usize>,
}

pub trait Layer {
  fn load_onnx_layer(config: &LayerConfig) -> (Vec<usize>, Vec<Vec<(i32, usize)>>); // vec of basic block idx & vec of inputs
  
  fn consume_basic_block(config: &LayerConfig) -> Vec<Box<dyn BasicBlock>>;
  
  fn layer_output_node(config: &LayerConfig) -> (usize, usize);
}
