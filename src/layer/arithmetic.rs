use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

macro_rules! define_arithmetic_layer {
  ($struct_name:ident, $basic_block:ident) => {
    pub struct $struct_name;

    impl Layer for $struct_name {
      fn graph(
        input_shapes: &Vec<&Vec<usize>>,
        input_types: &Vec<DatumType>,
        _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
        _attributes: &Vec<&AttributeProto>,
      ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
        let mut graph = Graph::new();
        let layer = if input_shapes[0].len() == 0 && input_shapes[1].len() == 0 {
          graph.addBB(Box::new($basic_block {}))
        } else {
          graph.addBB(Box::new(RepeaterBasicBlock {
            basic_block: Box::new($basic_block {}),
            N: 1,
          }))
        };
        let output_shape = if input_shapes[0].len() == 0 && input_shapes[1].len() == 0 {
          input_shapes[0].clone()
        } else {
          util::broadcastDims(input_shapes, 0)
        };
        let layer_output = graph.addNode(layer, vec![(-1, 0), (-2, 0)]);
        graph.outputs.push((layer_output, 0));
        (graph, vec![output_shape], vec![input_types[0]])
      }
    }
  };
}

// Using the macro to define AddLayer and SubLayer
define_arithmetic_layer!(AddLayer, AddBasicBlock);
define_arithmetic_layer!(SubLayer, SubBasicBlock);
