use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct SqueezeLayer;
impl Layer for SqueezeLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axes").next().unwrap().ints[0] as isize;
    let axis = if axis < 0 { input_shapes[0].len() as isize + axis } else { axis };

    let startShape = input_shapes[0];
    assert!(startShape[axis as usize] == 1);
    let endShape: Vec<_> = startShape.iter().enumerate().filter(|(i, _)| *i != axis as usize).map(|(_, x)| *x).collect();
    
    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: endShape.clone() }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else { // startShape.last() < endShape.last()
      let n = startShape.len();
      let (mut a, mut b) = (startShape[n - 2], startShape[n - 1]);
      assert!(*endShape.last().unwrap() == a * b);
      (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
      let permutation = (vec![0], (0..a * b).collect());
      println!("{:?}", a*b);
      println!("{:?}", permutation);
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let intermediateShape = endShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: intermediateShape }));
      let intermediate = graph.addNode(permute, vec![(-1, 0)]);
      let output = graph.addNode(reshape, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape])
  }
}

pub struct UnsqueezeLayer;
impl Layer for UnsqueezeLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axes").next().unwrap().ints[0] as isize;
    let axis = if axis < 0 { input_shapes[0].len() as isize + axis + 1 } else { axis };

    let startShape = input_shapes[0];
    let endShape: Vec<_> = (0..startShape.len() + 1).map(|x| if x == axis as usize { 1 } else { if x > axis as usize { startShape[x-1] } else { startShape[x] } }).collect();

    if startShape.last() == endShape.last() {
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: endShape.clone() }));
      let output = graph.addNode(reshape, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else if startShape.last() > endShape.last() {
      let n = endShape.len();
      let (mut a, mut b) = (endShape[n - 2], endShape[n - 1]);
      assert!(*startShape.last().unwrap() == a * b);
      let mut intermediateShape = endShape[..n - 2].to_vec();
      intermediateShape.push(1);
      intermediateShape.push(*startShape.last().unwrap());
      intermediateShape.iter_mut().for_each(|x| *x = util::next_pow(*x as u32) as usize);
      let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: intermediateShape }));
      (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
      let permutation = ((0..a).map(|x| x * b).collect(), (0..b).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let intermediate = graph.addNode(reshape, vec![(-1, 0)]);
      let output = graph.addNode(permute, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape])
  }
}
