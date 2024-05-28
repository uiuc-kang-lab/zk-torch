use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct CastLayer;
impl Layer for CastLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let id = graph.addBB(Box::new(IdBasicBlock {}));
    let id_output = graph.addNode(id, vec![(-1, 0)]);
    graph.outputs.push((id_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
