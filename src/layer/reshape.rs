use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct ReshapeLayer;
impl Layer for ReshapeLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let startShape = input_shapes[0];
    let endShape: Vec<_> = constants[1].unwrap().as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).filter(|x| *x != 0).collect();

    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: endShape.clone() }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else if startShape.last() > endShape.last() {
      let n = endShape.len();
      let (a, b) = (endShape[n - 2], endShape[n - 1]);
      assert!(*startShape.last().unwrap() == a * b);
      let mut intermediateShape = endShape[..n - 2].to_vec();
      intermediateShape.push(1);
      intermediateShape.push(*startShape.last().unwrap());
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: intermediateShape }));
      let permutation = ((0..a).map(|x| x * b).collect(), (0..b).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let intermediate = graph.addNode(reshape, vec![(-1, 0)]);
      let output = graph.addNode(permute, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    } else {
      let n = startShape.len();
      let (a, b) = (startShape[n - 2], startShape[n - 1]);
      assert!(*endShape.last().unwrap() == a * b);
      let permutation = (vec![0], (0..a * b).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: endShape.clone() }));
      let intermediate = graph.addNode(permute, vec![(-1, 0)]);
      let output = graph.addNode(reshape, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape])
  }
}
