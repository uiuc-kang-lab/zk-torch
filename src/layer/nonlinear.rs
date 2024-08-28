use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

macro_rules! define_nonlinear_layer {
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
        let layer = graph.addBB(Box::new($basic_block {
          input_SF: *onnx::SF_LOG,
          output_SF: *onnx::SF_LOG,
        }));
        let layer_check = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQ2BasicBlock {
            setup: Some((
              Box::new($basic_block {
                input_SF: *onnx::SF_LOG,
                output_SF: *onnx::SF_LOG,
              }),
              *onnx::CQ_RANGE_LOWER,
              *onnx::CQ_RANGE,
            )),
          }),
          N: 1,
        }));
        let layer_output = graph.addNode(layer, vec![(-1, 0)]);
        let _ = graph.addNode(layer_check, vec![(-1, 0), (layer_output, 0)]);
        graph.outputs.push((layer_output, 0));
        (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
      }
    }
  };
}

// Using the macro to define nonlinear layers
define_nonlinear_layer!(ReLULayer, ReLUBasicBlock);
define_nonlinear_layer!(SqrtLayer, SqrtBasicBlock);
define_nonlinear_layer!(CeilLayer, CeilBasicBlock);
define_nonlinear_layer!(ErfLayer, ErfBasicBlock);
define_nonlinear_layer!(ExpLayer, ExpBasicBlock);
define_nonlinear_layer!(SigmoidLayer, SigmoidBasicBlock);
define_nonlinear_layer!(TanhLayer, TanhBasicBlock);
define_nonlinear_layer!(CosLayer, CosBasicBlock);
define_nonlinear_layer!(SinLayer, SinBasicBlock);
define_nonlinear_layer!(TanLayer, TanBasicBlock);
define_nonlinear_layer!(ReciprocalLayer, ReciprocalBasicBlock);
