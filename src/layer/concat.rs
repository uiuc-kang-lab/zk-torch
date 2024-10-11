use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

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
      let input_shapes_clone: Vec<Vec<usize>> = input_shapes.clone().iter().map(|x| x.to_vec()).collect();
      let concat = graph.addBB(Box::new(ConcatLastDimBasicBlock {
        input_shapes: input_shapes_clone,
      }));
      let n_input = input_shapes.len();
      let concat_input: Vec<_> = (0..n_input).map(|i| (-(i as i32 + 1), 0)).collect();
      let concat_output = graph.addNode(concat, concat_input);
      graph.outputs.push((concat_output, 0));
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
