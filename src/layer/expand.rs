use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::One;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

#[derive(Debug)]
pub struct ExpandBasicBlock;
impl BasicBlock for ExpandBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let newShape: Vec<_> = inputs[1].as_slice().unwrap().iter().map(|&x| util::fr_to_int(x) as usize).filter(|x| *x != 0).collect();
    let padded_newShape: Vec<_> = newShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
    vec![inputs[0].broadcast(padded_newShape).unwrap().into_owned()]
  }
}

pub struct ExpandLayer;
impl Layer for ExpandLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let shape0 = input_shapes[0].clone();
    let shape1: Vec<_> = constants[1].unwrap().0.as_slice().unwrap().iter().map(|&x| util::fr_to_int(x) as usize).filter(|x| *x != 0).collect();
    let newShape = vec![shape0.clone(), shape1.clone()];
    let newShape: Vec<_> = newShape.iter().map(|x| x).collect();
    let newShape = util::broadcastDims(&newShape, 0);
    let newShape_padded: Vec<_> = newShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();

    let mut graph = Graph::new();
    // check if the last dimension of the input shape is equal to the last dimension of the new shape
    // and the product of the input shape is less than the product of the new shape
    // if so, use ExpandBasicBlock without proving. Otherwise, use ConstOfShapeBasicBlock and MulScalarBasicBlock
    let shape0_product = shape0.iter().fold(1, |acc, x| acc * x);
    let shape1_product = shape1.iter().fold(1, |acc, x| acc * x);
    if *input_shapes[0].last().unwrap() == *newShape.clone().last().unwrap() && shape0_product <= shape1_product {
      let expand = graph.addBB(Box::new(ExpandBasicBlock {}));
      let expand_output = graph.addNode(expand, vec![(-1, 0), (-2, 0)]);
      graph.outputs.push((expand_output, 0));
    } else {
      let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::one(),
        shape: newShape_padded.clone(),
      }));
      let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulScalarBasicBlock {}),
        N: 1,
      }));
      let mul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulBasicBlock {}),
        N: 1,
      }));
      let constantOfShape_output = graph.addNode(constantOfShape, vec![]);
      let expand_output = if *input_shapes[0].last().unwrap() == 1 {
        graph.addNode(mul_scalar, vec![(constantOfShape_output, 0), (-1, 0)])
      } else if *newShape_padded.last().unwrap() == 1 {
        graph.addNode(mul_scalar, vec![(-1, 0), (constantOfShape_output, 0)])
      } else {
        graph.addNode(mul, vec![(-1, 0), (constantOfShape_output, 0)])
      };
      graph.outputs.push((expand_output, 0));
    }

    (graph, vec![newShape], vec![input_types[0]])
  }
}
