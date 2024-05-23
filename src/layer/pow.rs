use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;

pub struct PowLayer;
impl Layer for PowLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    assert!(constants[1].unwrap().first().unwrap() == &Fr::from(2 * crate::onnx::SF as u32));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let pow_output = graph.addNode(mul, vec![(-1, 0), (-1, 0)]);
    graph.outputs.push((pow_output, 0));
    (graph, vec![input_shapes[0].clone()])
  }
}
