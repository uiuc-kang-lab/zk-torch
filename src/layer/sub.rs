use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1,ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct SubLayer;
impl Layer for SubLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let sub_output = graph.addNode(sub, vec![(-1, 0), (-2, 0)]);
    let finalShape = util::broadcastDims(input_shapes, 0);
    let lastDim = *finalShape.last().unwrap();
    let lastDimPow2 = util::next_pow(lastDim as u32) as usize;
    if input_shapes[0].last() == input_shapes[1].last(){
      graph.outputs.push((sub_output, 0));
    }else{
      println!("sub last dim {:?} {:?}",lastDim,lastDimPow2);
      let paddingConst:Vec<_> = std::iter::repeat(Fr::from(1)).take(lastDim).chain(std::iter::repeat(Fr::from(0)).take(lastDimPow2 - lastDim)).collect();
      let paddingConst = graph.addBB(Box::new(Const2BasicBlock { c: arr1(&paddingConst).into_dyn()}));
      let mul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulBasicBlock {}),
        N: 1,
      }));
      let padding_const = graph.addNode(paddingConst, vec![]);
      let mul_output = graph.addNode(mul, vec![(padding_const, 0), (sub_output, 0)]);
      graph.outputs.push((mul_output, 0));
    }
    (graph, vec![finalShape])
  }
}
