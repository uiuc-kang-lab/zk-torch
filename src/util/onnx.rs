use crate::onnx;
/*
 * ONNX utilities:
 * The function(s) are used for ONNX-related operations.
 * For example, generate fake inputs for ONNX models.
 */
use crate::util::pad_to_pow_of_two;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::ArrayD;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::prelude::{DatumType, Framework};

// This function is used for generating fake inputs for onnx models
// Fake inputs are random field (i.e., Fr) elements whose shapes and types match those described in the input tensors of an ONNX model.
// Generating these when loading an ONNX file saves us from creating different input tensors ourselves when testing new ONNX.
// It is only for testing purposes
pub fn generate_fake_inputs_for_onnx(filename: &str) -> Vec<ArrayD<Fr>> {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let mut inputs = vec![];

  for onnx_input in onnx_graph.input.iter() {
    let tract_onnx::pb::type_proto::Value::TensorType(t) = onnx_input.r#type.as_ref().unwrap().value.as_ref().unwrap();
    let shape = t
      .shape
      .as_ref()
      .unwrap()
      .dim
      .iter()
      .map(|x| {
        if let tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(x) = x.value.as_ref().unwrap() {
          *x as usize
        } else {
          panic!("Unknown dimension")
        }
      })
      .collect::<Vec<_>>();

    let input = generate_fake_tensor(t.elem_type(), shape);
    let input = pad_to_pow_of_two(&input, &Fr::zero());
    inputs.push(input);
  }
  inputs
}

pub fn generate_fake_tensor(dtype: DataType, shape: Vec<usize>) -> ArrayD<Fr> {
  eprintln!("\x1b[93mWARNING\x1b[0m: Generating fake tensor for ONNX model. This is only for testing purposes.");
  let mut rng = StdRng::from_entropy();
  let val_num = shape.iter().fold(1, |acc, x| acc * x);
  let input = match dtype {
    DataType::Float | DataType::Float16 | DataType::Double => (0..val_num).map(|_| Fr::from(rng.gen_range(-2..2))).collect(),
    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => (0..val_num).map(|_| Fr::from(1)).collect(),
    DataType::Uint8 | DataType::Uint16 | DataType::Uint32 | DataType::Uint64 => (0..val_num).map(|_| Fr::from(1)).collect(),
    DataType::Bool => (0..val_num).map(|_| Fr::from(rng.gen_range(0..2))).collect(),
    _ => panic!("Unsupported constant type: {:?}", dtype),
  };
  ArrayD::from_shape_vec(shape, input).unwrap()
}

// Converts ints for the DataType enum into DatumType
// https://docs.rs/tract-onnx/latest/tract_onnx/pb/tensor_proto/enum.DataType.html
pub fn datatype_to_datumtype(t: i32) -> DatumType {
  match t {
    2 | 3 | 4 | 5 | 6 | 7 | 12 | 13 => DatumType::I64,
    1 | 10 | 11 => DatumType::F32,
    8 => DatumType::String,
    9 => DatumType::Bool,
    _ => panic!("DatumType {:?} not supported", t),
  }
}

// Converts DatumType to the corresponding scale factor
// It should only be used in the IN_SF/OUT_SF of nonlinear basicblocks
pub fn datumtype_to_sf(t: DatumType) -> usize {
  match t {
    DatumType::I32 => 1,
    DatumType::I64 => 1,
    DatumType::Bool => 1,
    DatumType::F32 => *onnx::SF_LOG,
    _ => panic!("DatumType {:?} not supported", t),
  }
}
