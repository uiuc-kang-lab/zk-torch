use crate::basic_block::*;
use crate::graph::*;
use crate::layer;
use crate::layer::*;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use std::collections::HashMap;
use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::prelude::{DatumType, Framework};
use tract_onnx::tensor::load_tensor;

const SF: usize = 32;
const SF_LOG: usize = 5;
const SF_FLOAT: f32 = 32f32;

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
  let constants = onnx_graph.initializer.iter().take(10).map(|tensor| (tensor.name.clone(), tensor)).chain(
    onnx_graph
      .node
      .iter()
      .filter(|node| node.op_type == "Constant")
      .map(|node| (node.output[0].clone(), node.attribute[0].t.as_ref().unwrap())),
  );
  for (name, tensor) in constants {
    let tensor = load_tensor(&*onnx.provider, tensor, None).unwrap();
    let tensor = match tensor.datum_type() {
      DatumType::F32 => {
        let tensor = tensor.into_array::<f32>().unwrap();
        Ok(tensor.map(|x| Fr::from((*x * SF_FLOAT).round() as i32)))
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
    outputs_idx.insert(name, vec![(graph.basic_blocks.len() as i32, 0)]);
    graph.nodes.push(Node {
      basic_block: graph.basic_blocks.len(),
      inputs: vec![],
    });
    // We are currently have a ConstBasicBlock followed by a MatMulBasicBlock.
    // In the future, we can prune the Graph so that this is replaced by one CQLinBasicBlock.
    graph.basic_blocks.push(Box::new(ConstBasicBlock {}));
    setups.push(tensor);
    idx += 1;
  }

  let sizes: Vec<Option<Vec<(Vec<usize>, Vec<usize>)>>> = vec![None; onnx_graph.node.len()];
  for node in onnx_graph.node.iter().filter(|node| node.op_type.as_str() != "Constant").take(5) {
    let op = node.op_type.as_str();
    let input_shapes: Vec<_> = node.input.iter().map(|x| &shapes[x]).collect();
    let (mut local_graph, output_shapes) = match op {
      "Add" => Ok(AddLayer::graph(&input_shapes)),
      "Sub" => Ok(SubLayer::graph(&input_shapes)),
      "MatMul" => Ok(MatMulLayer::graph(&input_shapes)),
      "Relu" => Ok(ReLULayer::graph(&input_shapes)),
      "Gather" => Ok(GatherLayer::graph(&input_shapes)),
      "ReduceMean" => Ok(ReduceMeanLayer::graph(&input_shapes)),
      _ => Err(format!("Unsupported onnx operation: {op}")),
    }
    .unwrap();

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
    for local_node in local_graph.nodes {
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
    node.output.iter().zip(output_shapes).for_each(|(output, shape)| {
      shapes.insert(output.clone(), shape);
    });
  }

  (graph, setups)
}
