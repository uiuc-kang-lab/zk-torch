use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::Dimension;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Rotate the last dimension of the input tensor:
// [x_0, x_1, x_2, x_3, ..., x_{n-2}, x_{n-1}] -> [x_1, x_0, x_3, x_2, ..., x_{n-1}, x_{n-2}]
fn get_rope_rotate_indices(input_shape: &Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let indices = ArrayD::from_shape_fn(input_shape.as_slice(), |index| {
    let index_len = index.ndim();
    let index_last = index[index_len - 1];
    let new_index_last = 2 * (index_last / 2) + 1 - (index_last % 2);
    let mut index = index.clone();
    index[index_len - 1] = new_index_last;
    Some(index)
  });
  let indices = util::pad_to_pow_of_two(&indices, &None);
  indices
}

// Generate a tensor with a given value (the value is in the ONNX attribute) and shape (the shape is in the input tensor)
pub struct RopeConstLayer;
impl Layer for RopeConstLayer {
  fn graph(
    _input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let inputShape: Vec<usize> =
      constants[0].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).filter(|x| *x != 0).collect();
    let inputShape_last = inputShape[inputShape.len() - 1];
    // -1, 1, -1, 1, ...
    let rope_constant = (0..inputShape_last).map(|i| Fr::from((-1 as i32).pow((i + 1) as u32)).into()).collect();
    let mut endShape: Vec<usize> = (0..inputShape.len()).map(|_| 1).collect();
    let endShape_len = endShape.len();
    endShape[endShape_len - 1] = inputShape_last;
    let rope_constant_tensor = ArrayD::from_shape_vec(endShape.clone(), rope_constant).unwrap();
    let rope_constant_tensor = util::pad_to_pow_of_two(&rope_constant_tensor, &Fr::zero());

    let constant = graph.addBB(Box::new(Const2BasicBlock { c: rope_constant_tensor }));
    let output = graph.addNode(constant, vec![]);
    graph.outputs.push((output, 0));
    (graph, vec![endShape], vec![input_types[0]])
  }
}

pub struct RopeRotateLayer;
impl Layer for RopeRotateLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let inputShape = input_shapes[0].clone();
    let startShape_padded: Vec<usize> = inputShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();
    let permutation = get_rope_rotate_indices(&inputShape);
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation: permutation.clone(),
      input_dim: IxDyn(&startShape_padded),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((output, 0));
    (graph, vec![inputShape], vec![input_types[0]])
  }
}
