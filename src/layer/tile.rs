use crate::basic_block::*;
use crate::graph::*;
use crate::layer::{squeeze::UnsqueezeBasicBlock, Layer};
use crate::util;
use ark_bn254::Fr;
use ndarray::{concatenate, ArrayD, Axis, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Helper function to get the indices of the tiled tensor
fn get_tile_indices(input_shape: Vec<usize>, repeats: Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let output_shape: Vec<_> = input_shape.iter().zip(repeats.iter()).map(|(x, y)| x * y).collect();
  let padded_output_shape: Vec<_> = output_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();

  // first generate indices for the input tensor
  let mut tiled = ArrayD::from_shape_fn(input_shape.as_slice(), |index| Some(index.clone()));
  // then repeat the indices r time(s) along each axis, where r is the corresponding element in repeats
  for (i, repeat) in repeats.iter().enumerate() {
    tiled = concatenate(Axis(i), std::iter::repeat(tiled.view()).take(*repeat).collect::<Vec<_>>().as_slice()).unwrap();
  }
  assert!(tiled.shape() == output_shape.as_slice());
  // finally pad the tiled tensor to the next power of 2
  let padded_tiled = util::pad_to_pow_of_two(&tiled, &None);
  assert!(padded_tiled.shape() == padded_output_shape.as_slice());

  padded_tiled
}

// TileLayer is a layer that repeats the input tensor along each axis according to the repeats.
// The functionality is equivalent to numpy.tile(arr, repeats)
// reference: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
pub struct TileLayer;
impl Layer for TileLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let repeats: Vec<_> = constants[1].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x)).collect();
    let mut repeats: Vec<_> = repeats.iter().map(|x| *x as usize).collect();

    let mut input_shape = input_shapes[0].clone();
    let mut input_index = -1;

    // when the input.ndim() is shorter than repeats.len(), we need to unsqueeze the input
    if input_shape.len() < repeats.len() {
      // append 1s at the beginning of the input_shape
      let diff = repeats.len() - input_shape.len();
      input_shape = std::iter::repeat(1).take(diff).chain(input_shape.iter().cloned()).collect::<Vec<_>>();
      let unsq = graph.addBB(Box::new(UnsqueezeBasicBlock {}));
      input_index = graph.addNode(unsq, vec![(input_index, 0)]);
      for _ in 0..diff - 1 {
        input_index = graph.addNode(unsq, vec![(input_index, 0)]);
      }
    // when the input.ndim() is longer than repeats.len(), we need to pad the repeats
    } else if input_shape.len() > repeats.len() {
      // append 1s at the beginning of the repeats
      repeats = std::iter::repeat(1).take(input_shape.len() - repeats.len()).chain(repeats.iter().cloned()).collect();
    }
    // now input_shape.len() should be equal to repeats.len()
    assert!(input_shape.len() == repeats.len());
    let permutation = get_tile_indices(input_shape, repeats);

    let padded_output_shape = permutation.shape().to_vec();
    let padded_input_shape: Vec<_> = input_shapes[0].iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&padded_input_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let tiled_output = graph.addNode(cc, vec![(input_index, 0)]);
    graph.outputs.push((tiled_output, 0));

    (graph, vec![padded_output_shape], vec![input_types[0]])
  }
}
