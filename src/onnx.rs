use crate::basic_block::*;
use crate::graph::*;
use crate::layer::*;
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, ArrayD};
use std::collections::HashMap;
use tract_onnx::prelude::{DatumType, Framework};
use tract_onnx::tensor::load_tensor;

pub const SF_LOG: usize = 3; //9
pub const SF: usize = 1 << SF_LOG;
pub const SF_FLOAT: f32 = (1 << SF_LOG) as f32;
pub const CQ_RANGE: usize = 1 << 6; //12
pub const CQ_RANGE_LOWER: i32 = -(1 << 5);

pub fn load_file(filename: &str) -> (Graph, Vec<ArrayD<Fr>>) {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let mut input_idx = HashMap::new();
  let mut shapes = HashMap::new();
  for (idx, i) in onnx_graph.input.iter().enumerate() {
    input_idx.insert(i.name.clone(), idx);
    let tract_onnx::pb::type_proto::Value::TensorType(t) = i.r#type.as_ref().unwrap().value.as_ref().unwrap();
    shapes.insert(
      i.name.clone(),
      t.shape
        .as_ref()
        .unwrap()
        .dim
        .iter()
        .map(|x| {
          let tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(x) = x.value.as_ref().unwrap() else {
            panic!("unknown dimension")
          };
          *x as usize
        })
        .collect::<Vec<_>>(),
    );
  }

  let mut graph = Graph {
    basic_blocks: vec![],
    nodes: vec![],
    outputs: vec![],
  };
  let mut basic_blocks_idx: HashMap<String, usize> = HashMap::new(); // BasicBlock to graph.basic_blocks index
  let mut outputs_idx: HashMap<String, Vec<(i32, usize)>> = HashMap::new(); // Graph node name to graph.nodes outputs
  let mut setups = vec![];

  let mut idx = 0;
  let constants = onnx_graph.initializer.iter().map(|tensor| (tensor.name.clone(), tensor)).chain(
    onnx_graph
      .node
      .iter()
      .filter(|node| node.op_type == "Constant")
      .map(|node| (node.output[0].clone(), node.attribute[0].t.as_ref().unwrap())),
  );
  let mut constants_hashmap = HashMap::new();
  for (name, tensor) in constants {
    let tensor = load_tensor(&*onnx.provider, tensor, None).unwrap();
    let tensor = match tensor.datum_type() {
      DatumType::F32 => {
        let tensor = tensor.into_array::<f32>().unwrap();
        Ok(tensor.map(|x| {
          let mut y = (*x * SF_FLOAT).round();
          if y < -(1 << 15) as f32 {
            y = -(1 << 15) as f32;
          }
          if y > (1 << 15) as f32 {
            y = (1 << 15) as f32;
          }
          Fr::from(y as i32)
        }))
      }
      DatumType::I64 => {
        let tensor = tensor.into_array::<i64>().unwrap();
        Ok(tensor.map(|x| Fr::from(*x)))
      }
      _ => Err(format!("Unsupported constant type: {:?}", tensor.datum_type())),
    }
    .unwrap();
    shapes.insert(name.clone(), tensor.shape().to_vec());
    let tensor = util::pad(&tensor);
    outputs_idx.insert(name.clone(), vec![(graph.basic_blocks.len() as i32, 0)]);
    graph.nodes.push(Node {
      basic_block: graph.basic_blocks.len(),
      inputs: vec![],
    });
    // We are currently have a ConstBasicBlock followed by a MatMulBasicBlock.
    // In the future, we can prune the Graph so that this is replaced by one CQLinBasicBlock.
    graph.basic_blocks.push(Box::new(ConstBasicBlock {}));
    setups.push(tensor);
    constants_hashmap.insert(name, idx);
    idx += 1;
  }
  let mut passed_constants = HashMap::new();

  for node in onnx_graph.node.iter().filter(|node| node.op_type.as_str() != "Constant") {
    let op = node.op_type.as_str();
    let input_shapes: Vec<_> = node.input.iter().map(|x| &shapes[x]).collect();
    let my_constants = node.input.iter().map(|x| passed_constants.get(x).or(constants_hashmap.get(x).map(|&y| &setups[y]))).collect();
    let my_attributes = node.attribute.iter().map(|x| x).collect();
    let (mut local_graph, output_shapes) = match op {
      "Add" => Ok(AddLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Mul" => Ok(MulLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Cast" => Ok(CastLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Sub" => Ok(SubLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "MatMul" => Ok(MatMulLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Relu" => Ok(ReLULayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Gather" => Ok(GatherLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "ReduceMean" => Ok(ReduceMeanLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Pow" => Ok(PowLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Div" => Ok(DivLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Sqrt" => Ok(SqrtLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Reshape" => Ok(ReshapeLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Transpose" => Ok(TransposeLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Shape" => Ok(ShapeLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Equal" => Ok(EqualLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Where" => Ok(WhereLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Expand" => Ok(ExpandLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Softmax" => Ok(SoftmaxLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      "Erf" => Ok(ErfLayer::graph(&input_shapes, &my_constants, &my_attributes)),
      _ => Err(format!("Unsupported onnx operation: {op}")),
    }
    .unwrap();

    if my_constants.iter().all(|&x| x.is_some()) {
      let my_inputs: Vec<_> = my_constants.iter().map(|&x| x.unwrap().clone()).collect();
      let my_inputs = my_inputs.iter().map(|x| x).collect();
      let outputs = local_graph.run(&my_inputs, &vec![&arr1(&[]).into_dyn(); local_graph.basic_blocks.len()]);
      node.output.iter().zip(local_graph.outputs.iter()).for_each(|(output_str, &(nodeX, nodeY))| {
        passed_constants.insert(output_str, outputs[nodeX as usize][nodeY].clone());
      });
    }

    let mut local_block_idx = vec![];
    let temp = local_graph.basic_blocks;
    local_graph.basic_blocks = vec![];
    for basic_block in temp.into_iter() {
      let name = format!("{basic_block:?}");
      let idx = *basic_blocks_idx.entry(name).or_insert_with(|| graph.basic_blocks.len());
      local_block_idx.push(idx);
      if idx == graph.basic_blocks.len() {
        setups.push(basic_block.genModel());
        graph.basic_blocks.push(basic_block);
      }
    }
    let start_idx = graph.nodes.len() as i32;
    for local_node in local_graph.nodes.iter() {
      graph.nodes.push(Node {
        basic_block: local_block_idx[local_node.basic_block],
        inputs: local_node
          .inputs
          .iter()
          .map(|(x, y)| {
            if x < &0 {
              let input_tag = &node.input[(-x - 1) as usize];
              if input_idx.contains_key(input_tag) {
                (-(*input_idx.get(input_tag).unwrap() as i32) - 1, *y)
              } else {
                outputs_idx[input_tag][*y]
              }
            } else {
              (start_idx + *x, *y)
            }
          })
          .collect(),
      });
    }
    outputs_idx.insert(
      node.output[0].clone(),
      local_graph.outputs.iter().map(|(x, y)| (start_idx + x, *y)).collect(),
    );
    if op == "Shape" {
      passed_constants.insert(
        &node.output[0],
        arr1(&input_shapes[0].iter().map(|&x| Fr::from(x as i32)).collect::<Vec<_>>()).into_dyn(),
      );
    }
    node.output.iter().zip(output_shapes).for_each(|(output, shape)| {
      shapes.insert(output.clone(), shape);
    });
  }

  (graph, setups)
}
