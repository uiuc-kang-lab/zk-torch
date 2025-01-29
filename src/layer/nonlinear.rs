use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

macro_rules! define_nonlinear_layer {
  ($struct_name:ident, $basic_block:ident, $enum_name:ident) => {
    pub struct $struct_name;

    impl Layer for $struct_name {
      fn graph(
        input_shapes: &Vec<&Vec<usize>>,
        input_types: &Vec<DatumType>,
        _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
        _attributes: &Vec<&AttributeProto>,
      ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
        let mut graph = Graph::new();
        let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
        let layer = graph.addBB(Box::new($basic_block {
          input_SF: sf_log,
          output_SF: sf_log,
        }));
        let layer_check = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQ2BasicBlock {
            op: cq2::CQ2BasicBlockOps::$enum_name(sf_log, sf_log),
            offset: *onnx::CQ_RANGE_LOWER,
            size: *onnx::CQ_RANGE,
            n: input_shapes[0][input_shapes[0].len() - 1].next_power_of_two(),
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
define_nonlinear_layer!(ReLULayer, ReLUBasicBlock, ReLU);
define_nonlinear_layer!(CeilLayer, CeilBasicBlock, Ceil);
define_nonlinear_layer!(ErfLayer, ErfBasicBlock, Erf);
define_nonlinear_layer!(ExpLayer, ExpBasicBlock, Exp);
define_nonlinear_layer!(SigmoidLayer, SigmoidBasicBlock, Sigmoid);
define_nonlinear_layer!(TanhLayer, TanhBasicBlock, Tanh);
define_nonlinear_layer!(CosLayer, CosBasicBlock, Cos);
define_nonlinear_layer!(SinLayer, SinBasicBlock, Sin);
define_nonlinear_layer!(TanLayer, TanBasicBlock, Tan);
define_nonlinear_layer!(ReciprocalLayer, ReciprocalBasicBlock, Reciprocal);
define_nonlinear_layer!(GeLULayer, GeLUBasicBlock, GeLU);
