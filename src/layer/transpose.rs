use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct TransposeLayer;
impl Layer for TransposeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let axes: Vec<_> = attributes.iter().filter(|x| x.name == "perm").next().unwrap().ints.iter().map(|x| *x as usize).collect();
    let n = axes.len();
    let endShape = axes.iter().map(|i| input_shapes[0][*i]).collect();

    if *axes.last().unwrap() == n - 1 {
      let transpose = graph.addBB(Box::new(TransposeBasicBlock { perm: axes.clone() }));
      let output = graph.addNode(transpose, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else {
      // move last element to be 2nd to last
      let pos = axes.iter().position(|&x| x == n - 1).unwrap();
      let mut perm = axes.clone();
      perm[pos] = axes[n - 2];
      perm[n - 2] = n - 1;
      let transpose = graph.addBB(Box::new(TransposeBasicBlock { perm }));
      let intermediate = graph.addNode(transpose, vec![(-1, 0)]);
      // swap the last two
      let (a, b) = (n - 1, axes[n - 1]);
      let (mut c, mut d) = (input_shapes[0][a], input_shapes[0][b]);
      c = util::next_pow(c as u32) as usize;
      d = util::next_pow(d as u32) as usize;
      let permutation = ((0..c).map(|x| x * d).collect(), (0..d).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation }),
        N: 2,
      }));
      let permute_output = graph.addNode(permute, vec![(intermediate, 0)]);
      // if pos swap happened, correct the swap
      let output = if pos == n - 2 {
        permute_output
      } else {
        let mut swap: Vec<_> = (0..n).collect();
        swap[pos] = n - 2;
        swap[n - 2] = pos;
        let transpose_1 = graph.addBB(Box::new(TransposeBasicBlock { perm: swap }));
        graph.addNode(transpose_1, vec![(permute_output, 0)])
      };
      graph.outputs.push((output, 0));
    }

    (graph, vec![endShape], vec![input_types[0]])
  }
}
