use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

// Generate a tensor with a given value (the value is in the ONNX attribute) and shape (the shape is in the input tensor)
// reference: https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
pub struct ConstOfShapeLayer;
impl Layer for ConstOfShapeLayer {
  fn graph(_input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let value = Fr::from(attributes.iter().filter(|x| x.name == "value").next().unwrap().i);
    let endShape: Vec<usize> = constants[0].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).filter(|x| *x != 0).collect();

    let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
      c: value,
      shape: endShape.clone(),
    }));
    let output = graph.addNode(constantOfShape, vec![]);
    graph.outputs.push((output, 0));
    (graph, vec![endShape])
  }
}
