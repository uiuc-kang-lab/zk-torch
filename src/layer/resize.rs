use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::Dimension;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Assumes nearest_mode = floor
fn resize_permutation(input_shape: &Vec<usize>, scales: &Vec<f32>) -> (Vec<usize>, ArrayD<Option<IxDyn>>) {
  let output_shape: Vec<_> = input_shape.iter().zip(scales).map(|(dim, scale)| (*dim as f32 * scale) as usize).collect();
  (
    output_shape.clone(),
    ArrayD::from_shape_fn(output_shape, |index| {
      let new_index: Vec<_> = index.as_array_view().to_vec().iter().enumerate().map(|(i, x)| (*x as f32 / scales[i]) as usize).collect();
      Some(IxDyn(&new_index))
    }),
  )
}

pub struct ResizeLayer;
// Only supports coordinate_transformation_mode = asymmetric, mode = nearest, and nearest_mode = floor
impl Layer for ResizeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let ctm = match attributes.iter().filter(|x| x.name == "coordinate_transformation_mode").next() {
      Some(v) => v.s.clone(),
      None => vec![],
    };
    let _ctm = if let Ok(str) = String::from_utf8(ctm) {
      if str == "asymmetric" {
        str
      } else {
        panic!("Resize only supports coordinate_transformation_mode = asymmetric");
      }
    } else {
      panic!("Resize has invalid coordinate_transformation_mode string");
    };
    let mode = match attributes.iter().filter(|x| x.name == "mode").next() {
      Some(v) => v.s.clone(),
      None => vec![],
    };
    let _mode = if let Ok(str) = String::from_utf8(mode) {
      if str == "nearest" {
        str
      } else {
        panic!("Resize only supports mode = nearest");
      }
    } else {
      panic!("Resize has invalid mode string");
    };
    let nearest_mode = match attributes.iter().filter(|x| x.name == "nearest_mode").next() {
      Some(v) => v.s.clone(),
      None => vec![],
    };
    let _nearest_mode = if let Ok(str) = String::from_utf8(nearest_mode) {
      if str == "floor" {
        str
      } else {
        panic!("Resize only supports nearest_mode = floor");
      }
    } else {
      panic!("Resize has invalid nearest_mode string");
    };
    // scales is a float, so it will have been scaled by onnx::SF by the onnx compiler. This assumes that dividing by onnx::SF will recover the original value
    let scales: Vec<_> = constants[2].unwrap().0.iter().map(|x| util::fr_to_int(*x) as f32 / *onnx::SF as f32).collect();
    assert!(scales.len() == input_shapes[0].len());

    let (output_shape, permutation) = resize_permutation(input_shapes[0], &scales);
    let permutation = util::pad_to_pow_of_two(&permutation, &None);

    let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&input_shape_padded),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));
    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((cc_output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
