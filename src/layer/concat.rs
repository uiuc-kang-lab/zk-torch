use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct ConcatLayer;
impl Layer for ConcatLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = if axis < 0 { input_shapes[0].len() as isize + axis } else { axis };

    let n_input = input_shapes.len();
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: axis as usize }));
    let concat_output = graph.addNode(concat, (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect());
    graph.outputs.push((concat_output, 0));

    let mut outputShape = input_shapes[0].clone();
    outputShape[axis as usize] = input_shapes.iter().map(|x| x[axis as usize]).sum();

    (graph, vec![outputShape])
  }
}