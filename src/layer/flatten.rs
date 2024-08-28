use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::util;
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

fn get_permutation(input_shape: &[usize], axis: usize) -> (ArrayD<Option<IxDyn>>, Vec<usize>) {
  assert!(axis < input_shape.len());
  let output_shape = if axis == 0 {
    vec![1, input_shape.iter().product()]
  } else {
    vec![input_shape[..axis].iter().product(), input_shape[axis..].iter().product()]
  };

  let permutation = ArrayD::from_shape_fn(input_shape, |index| Some(index));
  let permutation = permutation.view().into_shape(&output_shape.clone()[..]).unwrap().to_owned();
  let padded_permutation = util::pad_to_pow_of_two(&permutation, &None);

  (padded_permutation, output_shape)
}

// https://onnx.ai/onnx/operators/onnx__Flatten.html
// Flattens the input tensor into a 2D matrix.
// If input tensor has shape (d_0, d_1, ..., d_n) then the output will have shape (d_0 × d_1 × ... × d_{axis-1}, d_{axis} × d_{axis+1} × ... × dn).
pub struct FlattenLayer;
impl Layer for FlattenLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let axis: isize = attributes.iter().filter(|x| x.name == "axis").next().unwrap().i as isize;
    let axis = (if axis < 0 { input_shapes[0].len() as isize + axis } else { axis }) as usize;

    let padded_input_shape: Vec<usize> = input_shapes[0].iter().map(|&x| util::next_pow(x as u32) as usize).collect();

    let (permutation, output_shape) = get_permutation(&input_shapes[0], axis);

    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation,
      input_dim: IxDyn(&padded_input_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
