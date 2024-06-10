use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;

fn get_concat_indices(input_shapes:  &Vec<&Vec<usize>>, output_shape: &Vec<usize>, axis: usize) -> Vec<ArrayD<Option<IxDyn>>> {
  let mut indices = vec![];
  let mut axis_offset = 0;
  for i in 0..input_shapes.len() {
    let output = ArrayD::from_shape_fn(output_shape.as_slice(), |index| {
      if index[axis] >= axis_offset && index[axis] < axis_offset + input_shapes[i][axis] {
        let mut new_index = index.clone();
        new_index[axis] = index[axis] - axis_offset;
        Some(new_index)
      } else {
        None
      }
    });
    axis_offset += input_shapes[i][axis];
    indices.push(output);
  }
  indices
}

pub struct ConcatLayer;
impl Layer for ConcatLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;
    let mut outputShape = input_shapes[0].clone();
    outputShape[axis] = input_shapes.iter().map(|x| x[axis as usize]).sum();
    if axis == input_shapes[0].len() - 1 {
      let mut padded_output_shape = outputShape.clone();
      padded_output_shape[axis] = util::next_pow(padded_output_shape[axis] as u32) as usize;
      let permutations = get_concat_indices(input_shapes, &padded_output_shape, axis);
      let mut cc_basicblocks = vec![];
      for i in 0..input_shapes.len() {
        let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
          permutation: permutations[i].clone(),
          input_dim: IxDyn(&input_shapes[i]),
        }));
        cc_basicblocks.push(cc);
      }
      let add = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(AddBasicBlock {}),
        N: 1,
      }));
      
      let mut cc_outputs = vec![];
      for i in 0..input_shapes.len() {
        let cc_output = graph.addNode(cc_basicblocks[i], vec![(-(i as i32 + 1), 0)]);
        cc_outputs.push((cc_output, 0));
      }
      // add 2 cc_outputs and reduce to 1 output until only 1 output left
      while cc_outputs.len() > 1 {
        let add_output = graph.addNode(add, vec![cc_outputs.pop().unwrap(), cc_outputs.pop().unwrap()]);
        cc_outputs.push((add_output, 0));
      }
      let final_output = cc_outputs.pop().unwrap();
      graph.outputs.push(final_output);
    } else {
      let n_input = input_shapes.len();
      let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: axis as usize }));
      let concat_output = graph.addNode(concat, (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect());
      graph.outputs.push((concat_output, 0));
    }

    (graph, vec![outputShape])
  }
}
