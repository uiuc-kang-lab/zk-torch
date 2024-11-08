use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, Axis, Dimension, IxDyn};
use std::error::Error;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

fn gather_along_axis(input: &ArrayD<Fr>, indices: &ArrayD<Fr>, axis: usize) -> Result<ArrayD<Fr>, Box<dyn Error>> {
  // Validate axis
  if axis >= input.ndim() {
    return Err("Axis out of bounds".into());
  }

  // Compute the output shape
  let mut output_shape = input.shape().to_vec();
  // Replace the dimension at 'axis' with the shape of 'indices'
  output_shape.splice(axis..axis + 1, indices.shape().iter().cloned());
  let output_dim = IxDyn(&output_shape);

  // Create the output array
  let mut output = ArrayD::<Fr>::zeros(output_dim);

  // Iterate over the output indices
  for (out_idx, out_elem) in output.indexed_iter_mut() {
    // Build the corresponding input index
    let mut input_idx = Vec::with_capacity(input.ndim());
    let mut indices_idx = Vec::with_capacity(indices.ndim());
    let mut out_dim_iter = out_idx.slice().iter();

    for i in 0..input.ndim() {
      if i == axis {
        // For the axis we're gathering along, collect indices dimensions
        for _ in 0..indices.ndim() {
          let idx = *out_dim_iter.next().unwrap();
          indices_idx.push(idx);
        }
        let idx = indices[IxDyn(&indices_idx)];
        let idx = util::fr_to_int(idx) as usize;
        if idx >= input.shape()[axis] {
          return Err("Index out of bounds".into());
        }
        input_idx.push(idx);
      } else {
        let idx = *out_dim_iter.next().unwrap();
        input_idx.push(idx);
      }
    }

    // Assign the value from 'input' to 'output'
    *out_elem = input[IxDyn(&input_idx)];
  }

  Ok(output)
}

#[derive(Debug)]
pub struct GatherBasicBlock {
  pub axis: usize,
}

impl BasicBlock for GatherBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let v = gather_along_axis(inputs[0], inputs[1], self.axis).unwrap();
    Ok(vec![v])
  }
}

pub struct GatherLayer;
impl Layer for GatherLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;

    let mut indices_output = -2;
    // Avoid unwrapping a None value
    if input_shapes[1].len() == 0 {
      let indices = constants[1].unwrap().0.mapv(|x| {
        if x > Fr::from(input_shapes[0][axis] as i64) {
          Fr::from(input_shapes[0][axis] as i64) + x
        } else {
          x
        }
      });
      let indices = graph.addBB(Box::new(Const2BasicBlock { c: indices }));
      indices_output = graph.addNode(indices, vec![]);
    }
    let gather = graph.addBB(Box::new(GatherBasicBlock { axis: axis }));
    let output = graph.addNode(gather, vec![(-1, 0), (indices_output, 0)]);
    graph.outputs.push((output, 0));
    let mut output_shape = input_shapes[0].to_vec();
    output_shape.splice(axis..axis + 1, input_shapes[1].iter().cloned());
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
