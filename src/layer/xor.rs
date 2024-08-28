use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct XorLayer;
impl Layer for XorLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let bool_check = graph.addBB(Box::new(BooleanCheckBasicBlock {}));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let _ = graph.addNode(bool_check, vec![(-1, 0)]);
    let _ = graph.addNode(bool_check, vec![(-2, 0)]);
    let sub_output = graph.addNode(sub, vec![(-1, 0), (-2, 0)]);
    let xor_output = graph.addNode(mul, vec![(sub_output, 0), (sub_output, 0)]); // XOR(a, b) = PointwiseMul((a - b), (a - b))
    graph.outputs.push((xor_output, 0));
    (graph, vec![util::broadcastDims(input_shapes, 0)], vec![DatumType::Bool])
  }
}
