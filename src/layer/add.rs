use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1,ArrayD};
use tract_onnx::pb::AttributeProto;

pub struct AddLayer;
impl Layer for AddLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let add_output = graph.addNode(add, vec![(-1, 0), (-2, 0)]);
    let finalShape = util::broadcastDims(input_shapes, 0);
    let lastDim = *finalShape.last().unwrap();
    let lastDimPow2 = util::next_pow(lastDim as u32) as usize;
    if input_shapes[0].last() == input_shapes[1].last(){
      graph.outputs.push((add_output, 0));
    }else{
      println!("add last dim {:?} {:?}",lastDim,lastDimPow2);
      let paddingConst:Vec<_> = std::iter::repeat(Fr::from(1)).take(lastDim).chain(std::iter::repeat(Fr::from(0)).take(lastDimPow2 - lastDim)).collect();
      let paddingConst = graph.addBB(Box::new(Const2BasicBlock { c: arr1(&paddingConst).into_dyn()}));
      let mul = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MulBasicBlock {}),
        N: 1,
      }));
      let padding_const = graph.addNode(paddingConst, vec![]);
      let mul_output = graph.addNode(mul, vec![(padding_const, 0), (add_output, 0)]);
      graph.outputs.push((mul_output, 0));
    }
    (graph, vec![finalShape])
  }
}
