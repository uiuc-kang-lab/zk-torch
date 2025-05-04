use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_bn254::G1Projective;
use ark_std::Zero;
use ndarray::Dimension;
use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Assumes nearest_mode = floor
fn resize_permutation(input_shape: &Vec<usize>, sizes: &Vec<usize>) -> ArrayD<Option<IxDyn>> {
  let scales: Vec<_> = input_shape.iter().zip(sizes).map(|(dim, size)| *size as f32 / *dim as f32).collect();
  ArrayD::from_shape_fn(sizes.clone(), |index| {
    let new_index: Vec<_> = index.as_array_view().to_vec().iter().enumerate().map(|(i, x)| (*x as f32 / scales[i]) as usize).collect();
    Some(IxDyn(&new_index))
  })
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

    let sizes: Vec<_> = match constants.get(3) {
      Some(x) => x.unwrap().0.iter().map(|x| util::fr_to_int(*x) as usize).collect(),
      None => {
        let scales: Vec<_> = match constants.get(2) {
          Some(x) => x.unwrap().0.iter().map(|x| util::fr_to_int(*x) as f32 / onnx::SF.read().unwrap().to_owned() as f32).collect(),
          None => panic!("Both Resize sizes and scales constants are None"),
        };
        let output_shape: Vec<_> = input_shapes[0].iter().zip(scales).map(|(dim, scale)| (*dim as f32 * scale) as usize).collect();
        output_shape
      }
    };

    assert!(sizes.len() == input_shapes[0].len());

    let output_shape = sizes.clone();
    let permutation = resize_permutation(input_shapes[0], &sizes);
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

#[derive(Debug)]
pub struct CustomResizeBasicBlock {
  pub input_shape: Vec<usize>,  // [1, H_in * W_in, C]
  pub output_shape: Vec<usize>, // [1, H_out * W_out, C]
}
impl BasicBlock for CustomResizeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let input = inputs[0].to_owned();
    let C = self.input_shape[2];
    let D = (self.input_shape[1] as f64).sqrt() as usize;
    let D_o = (self.output_shape[1] as f64).sqrt() as usize;
    let scale = (D_o as f64 / D as f64) as usize;

    let mut result = ArrayD::<Fr>::zeros(IxDyn(&[1, D_o * D_o, C]));
    for i in 0..D_o {
      for j in 0..D_o {
        for c in 0..C {
          result[[0, i * D_o + j, c]] = input[[0, (i / scale) * D + (j / scale), c]];
        }
      }
    }
    result = util::pad_to_pow_of_two(&result, &Fr::zero());

    Ok(vec![result])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let input = inputs[0];
    let C = self.input_shape[2];
    let D = (self.input_shape[1] as f64).sqrt() as usize;
    let D_o = (self.output_shape[1] as f64).sqrt() as usize;
    let scale = (D_o as f64 / D as f64) as usize;
    let data_zero = Data {
      raw: vec![Fr::zero(); C],
      poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
      r: Fr::zero(),
      g1: G1Projective::zero(),
    };
    let mut result = ArrayD::from_shape_fn(IxDyn(&[1, D_o * D_o]), |_| data_zero.clone());
    for i in 0..D_o {
      for j in 0..D_o {
        result[[0, i * D_o + j]] = input[[0, (i / scale) * D + (j / scale)]].clone();
      }
    }
    result = util::pad_to_pow_of_two(&result, &data_zero);
    vec![result]
  }
}

pub struct CustomResizeLayer;
impl Layer for CustomResizeLayer {
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

    let sizes: Vec<_> = match constants.get(3) {
      Some(x) => x.unwrap().0.iter().map(|x| util::fr_to_int(*x) as usize).collect(),
      None => {
        let scales: Vec<_> = match constants.get(2) {
          Some(x) => x.unwrap().0.iter().map(|x| util::fr_to_int(*x) as f32 / onnx::SF.read().unwrap().to_owned() as f32).collect(),
          None => panic!("Both Resize sizes and scales constants are None"),
        };
        let output_shape: Vec<_> = input_shapes[0].iter().zip(scales).map(|(dim, scale)| (*dim as f32 * scale) as usize).collect();
        output_shape
      }
    };
    let sizes: Vec<_> = sizes.iter().map(|&x| x).filter(|x| *x != 0).collect();

    assert!(sizes.len() == input_shapes[0].len());

    let output_shape = sizes.clone();
    let cc = graph.addBB(Box::new(CustomResizeBasicBlock {
      input_shape: input_shapes[0].to_vec(),
      output_shape: output_shape.clone(),
    }));
    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    graph.outputs.push((cc_output, 0));

    (graph, vec![output_shape], vec![input_types[0]])
  }
}
