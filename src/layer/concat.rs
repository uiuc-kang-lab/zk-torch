use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;

pub struct ConcatLayer;
impl Layer for ConcatLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;
    let mut outputShape = input_shapes[0].clone();
    outputShape[axis] = input_shapes.iter().map(|x| x[axis as usize]).sum();
    if input_shapes[0].len() == 1 {
      // workaround: for 1D input. Maybe we should use copy constraint to handle this case later.
      let n_input = input_shapes.len();
      let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: axis as usize }));
      let concat_output = graph.addNode(concat, (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect());
      graph.outputs.push((concat_output, 0));
    } else if axis == input_shapes[0].len() - 1 {
      // permute inputs
      let n = outputShape.len();
      let mut a = outputShape[n - 2];
      let mut b = outputShape[n - 1];
      (a, b) = (util::next_pow(a as u32) as usize, util::next_pow(b as u32) as usize);
      let permutation = (vec![0], (0..a).map(|x| x).collect());
      let permute = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
        N: 2,
      }));
      let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: (axis - 1) as usize }));
      let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
      let permute_back = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(PermuteBasicBlock {
          permutation: permutation_back,
        }),
        N: 2,
      }));

      let n_input = input_shapes.len();
      let mut n_permute_output = Vec::with_capacity(n_input);
      for i in 0..n_input {
        let permute_output = graph.addNode(permute, vec![(-(i as i32 + 1), 0)]);
        n_permute_output.push((permute_output, 0));
      }
      let concat_output = graph.addNode(concat, n_permute_output);
      let output = graph.addNode(permute_back, vec![(concat_output, 0)]);
      graph.outputs.push((output, 0));
    } else {
      let n_input = input_shapes.len();
      let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: axis as usize }));
      let concat_output = graph.addNode(concat, (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect());
      graph.outputs.push((concat_output, 0));
    }

    (graph, vec![outputShape])
  }
}
