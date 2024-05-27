use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct TransposeLayer;
impl Layer for TransposeLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axes: Vec<_> = attributes.iter().filter(|x| x.name == "perm").next().unwrap().ints.iter().map(|x| *x as usize).collect();
    let n = axes.len();
    let endShape = axes.iter().map(|i| input_shapes[0][*i]).collect();

    if *axes.last().unwrap() == n - 1 {
      let transpose = graph.addBB(Box::new(TransposeBasicBlock { perm: axes.clone() }));
      let output = graph.addNode(transpose, vec![(-1, 0)]);
      graph.outputs.push((output, 0));
    } else if axes[n - 2] == n - 1 {
      let (a, b) = (axes[n - 2], axes[n - 1]); //3,1
      let mut perm = axes[..n - 2].to_vec();
      perm.push(b);
      perm.push(a);
      let transpose = graph.addBB(Box::new(TransposeBasicBlock { perm: perm }));
      let intermediate = graph.addNode(transpose, vec![(-1, 0)]);
      let (mut c, mut d) = (input_shapes[0][a], input_shapes[0][b]);
      c = util::next_pow(c as u32) as usize;
      d = util::next_pow(d as u32) as usize;
      let permutation = ((0..c).map(|x| x * d).collect(), (0..d).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let output = graph.addNode(permute, vec![(intermediate, 0)]);
      graph.outputs.push((output, 0));
    } else {
      todo!()
    }

    (graph, vec![endShape])
  }
}
