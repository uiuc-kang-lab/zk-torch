use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// This function returns N outputs where N is the number of inputs.
// Each output is an array with the same shape as the final concatenation array.
// And the value at each index is the index of the corresponding input array.
// For example, [1], [1], [1] -> [0, None, None], [None, 0, None], [None, None, 0]
// such that we can use the indices to copy the input arrays to a padded array and add them together into the final output array.
fn get_concat_indices(input_shapes: &Vec<&Vec<usize>>, output_shape: &Vec<usize>, axis: usize) -> Vec<ArrayD<Option<IxDyn>>> {
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
    let output = util::pad_to_pow_of_two(&output, &None);
    indices.push(output);
  }
  indices
}

// Concatenate the input arrays along the specified axis.
// If the axis is the last axis, we copy the input arrays to a padded array by Copy Constraint and add them together.
// Otherwise, we directly concatenate the input arrays.
pub struct ConcatLayer;
impl Layer for ConcatLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    // Extract the 'axis' attribute and adjust for negative values
    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;
    // Compute the output shape after concatenation
    let mut outputShape = input_shapes[0].clone();
    outputShape[axis] = input_shapes.iter().map(|x| x[axis as usize]).sum();
    // If concatenating along the last axis, use copy constraint as the output commitment changes
    if axis == input_shapes[0].len() - 1 {
      let mut padded_output_shape = outputShape.clone();
      padded_output_shape[axis] = util::next_pow(padded_output_shape[axis] as u32) as usize;
      let permutations = get_concat_indices(input_shapes, &padded_output_shape, axis);
      let mut cc_basicblocks = vec![];
      for i in 0..input_shapes.len() {
        let padded_input_shape: Vec<usize> = input_shapes[i].iter().map(|&x| util::next_pow(x as u32) as usize).collect();
        let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
          permutation: permutations[i].clone(),
          input_dim: IxDyn(&padded_input_shape),
          padding_partition: copy_constraint::PaddingEnum::Zero,
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
      // add 2 cc_outputs at a time until only 1 output is left
      while cc_outputs.len() > 1 {
        let add_output = graph.addNode(add, vec![cc_outputs.pop().unwrap(), cc_outputs.pop().unwrap()]);
        cc_outputs.push((add_output, 0));
      }
      let final_output = cc_outputs.pop().unwrap();
      graph.outputs.push(final_output);
    } else {
      // If not concatenating along the last axis, directly concatenate
      let mut constOfShape_shape = input_shapes[0].clone();
      constOfShape_shape[axis] = 1;
      let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::zero(),
        shape: constOfShape_shape.iter().map(|&x| util::next_pow(x as u32) as usize).collect(),
      }));
      let constantOfShape_output = graph.addNode(constantOfShape, vec![]);

      let n_input = input_shapes.len();
      let n_input_padded = util::next_pow(n_input as u32) as usize;

      let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: axis as usize }));
      let mut concat_input: Vec<_> = (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect();
      for _ in 0..n_input_padded - n_input {
        concat_input.push((constantOfShape_output, 0));
      }
      let concat_output = graph.addNode(concat, concat_input);
      graph.outputs.push((concat_output, 0));
    }

    (graph, vec![outputShape], vec![input_types[0]])
  }
}
