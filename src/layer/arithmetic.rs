use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

macro_rules! define_arithmetic_layer {
  ($struct_name:ident, $basic_block:ident) => {
      pub struct $struct_name;

      impl Layer for $struct_name {
          fn graph(
              input_shapes: &Vec<&Vec<usize>>,
              _constants: &Vec<Option<&ArrayD<Fr>>>,
              _attributes: &Vec<&AttributeProto>
          ) -> (Graph, Vec<Vec<usize>>) {
              let mut graph = Graph::new();
              let layer = graph.addBB(Box::new(RepeaterBasicBlock {
                  basic_block: Box::new($basic_block {}),
                  N: 1,
              }));
              let layer_output = graph.addNode(layer, vec![(-1, 0), (-2, 0)]);
              graph.outputs.push((layer_output, 0));
              (graph, vec![util::broadcastDims(input_shapes, 0)])
          }
      }
  };
}

// Using the macro to define AddLayer and SubLayer
define_arithmetic_layer!(AddLayer, AddBasicBlock);
define_arithmetic_layer!(SubLayer, SubBasicBlock);
