use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

// Generate a tensor with a given value (the value is in the ONNX attribute) and shape (the shape is in the input tensor)
// reference: https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
pub struct ConstOfShapeLayer;
impl Layer for ConstOfShapeLayer {
  fn graph(
    _input_shapes: &Vec<&Vec<usize>>,
    _input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let attr_val = attributes.iter().filter(|x| x.name == "value").next().unwrap();
    let dtype = DataType::from_i32(attr_val.r#type).unwrap().into();
    let datum_type = match dtype {
      DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => DatumType::I64,
      DataType::Uint8 | DataType::Uint16 | DataType::Uint32 | DataType::Uint64 => DatumType::I64,
      DataType::Double | DataType::Float16 | DataType::Float => DatumType::F32,
      _ => panic!("Unsupported data type"),
    };
    let value = match datum_type {
      DatumType::I64 => Fr::from(attr_val.t.clone().unwrap().raw_data[0]),
      DatumType::F32 => Fr::from((attr_val.t.clone().unwrap().raw_data[0] as f32 * *onnx::SF_FLOAT).round() as i32),
      _ => panic!("Unsupported data type"),
    };
    let endShape: Vec<usize> = constants[0].unwrap().0.as_slice().unwrap().iter().map(|x| util::fr_to_int(*x) as usize).filter(|x| *x != 0).collect();
    let endShape_padded: Vec<usize> = endShape.iter().map(|&x| util::next_pow(x as u32) as usize).collect();

    let constantOfShape = graph.addBB(Box::new(ConstOfShapeBasicBlock {
      c: value,
      shape: endShape_padded.clone(),
    }));
    let output = graph.addNode(constantOfShape, vec![]);
    graph.outputs.push((output, 0));
    (graph, vec![endShape], vec![datum_type])
  }
}
