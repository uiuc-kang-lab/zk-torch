use crate::basic_block::*;
use crate::graph::*;
use crate::layer::*;
use crate::util;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{arr1, ArrayD};
use std::collections::HashMap;
use tract_onnx::pb;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::{DatumType, Framework};
use tract_onnx::tensor::load_tensor;

pub const SF_LOG: usize = 3; //9
pub const SF: usize = 1 << SF_LOG;
pub const SF_FLOAT: f32 = (1 << SF_LOG) as f32;
pub const CQ_RANGE: usize = 1 << 6; //12
pub const CQ_RANGE_LOWER: i32 = -(1 << 5);

// This function is used for parsing the inputs of onnx models
fn parse_onnx_inputs(onnx_graph: &pb::GraphProto) -> (HashMap<String, usize>, HashMap<String, Vec<usize>>) {
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
          if let tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(x) = x.value.as_ref().unwrap() {
            *x as usize
          } else {
            panic!("Unknown dimension") // we currently can only support constant dimensions
          }
        })
        .collect::<Vec<_>>(),
    );
  }

  (input_idx, shapes)
}

// This function is used for parsing the constants of onnx models
fn parse_onnx_constants<'a>(
  onnx_graph: &'a pb::GraphProto,
  shapes: &mut HashMap<String, Vec<usize>>,
) -> (
  impl Iterator<Item = (String, &'a pb::TensorProto)> + 'a,
  HashMap<String, usize>,
  Vec<ArrayD<Fr>>,
) {
  let onnx = tract_onnx::onnx();
  let constants = onnx_graph.initializer.iter().map(|tensor| (tensor.name.clone(), tensor)).chain(
    onnx_graph
      .node
      .iter()
      .filter(|node| node.op_type == "Constant")
      .map(|node| (node.output[0].clone(), node.attribute[0].t.as_ref().unwrap())),
  );

  let mut constants_hashmap = HashMap::new();
  let mut models = Vec::new();
  let mut idx = 0;

  for (name, tensor) in constants.clone() {
    let tensor = load_tensor(&*onnx.provider, tensor, None).unwrap();
    let tensor = match tensor.datum_type() {
      DatumType::F32 => {
        let tensor = tensor.into_array::<f32>().unwrap();
        Ok(tensor.map(|x| {
          let mut y = (*x * SF_FLOAT).round();
          y = y.clamp(-(1 << 15) as f32, (1 << 15) as f32);
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
    let tensor = util::pad_to_pow_of_two(&tensor, &Fr::zero());
    constants_hashmap.insert(name.clone(), idx);
    models.push(tensor);
    idx += 1;
  }

  (constants, constants_hashmap, models)
}

// This function is used for creating the output indices for constants of onnx models
fn create_output_indices<'a>(
  constants: impl Iterator<Item = (String, &'a pb::TensorProto)>,
  graph: &mut Graph,
) -> HashMap<String, Vec<(i32, usize)>> {
  let mut outputs_idx = HashMap::new();

  for (name, _) in constants {
    outputs_idx.insert(name.clone(), vec![(graph.basic_blocks.len() as i32, 0)]);
    graph.nodes.push(Node {
      basic_block: graph.basic_blocks.len(),
      inputs: vec![],
    });
    graph.basic_blocks.push(Box::new(ConstBasicBlock {}));
  }

  outputs_idx
}

// This function is used for getting the local graph for each layer
fn get_local_graph(
  op: &str,
  input_shapes: &Vec<&Vec<usize>>,
  node_constants: &Vec<Option<&ArrayD<Fr>>>,
  node_attributes: Vec<&AttributeProto>,
) -> (Graph, Vec<Vec<usize>>) {
  match op {
    "Add" => Ok(AddLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Mul" => Ok(MulLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Cast" => Ok(CastLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Ceil" => Ok(CeilLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Concat" => Ok(ConcatLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "ConstantOfShape" => Ok(ConstOfShapeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Cos" => Ok(CosLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Sin" => Ok(SinLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Sub" => Ok(SubLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Einsum" => Ok(EinsumLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "LSTM" => Ok(LSTMLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "MatMul" => Ok(MatMulLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Neg" => Ok(NegLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Relu" => Ok(ReLULayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Gather" => Ok(GatherLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Range" => Ok(RangeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Reciprocal" => Ok(ReciprocalLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "ReduceMean" => Ok(ReduceMeanLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Pow" => Ok(PowLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Div" => Ok(DivLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "ScatterND" => Ok(ScatterNDLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Slice" => Ok(SliceLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Split" => Ok(SplitLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Sqrt" => Ok(SqrtLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Reshape" => Ok(ReshapeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Transpose" => Ok(TransposeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Tanh" => Ok(TanhLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Tile" => Ok(TileLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Shape" => Ok(ShapeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Sigmoid" => Ok(SigmoidLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Equal" => Ok(EqualLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Where" => Ok(WhereLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Expand" => Ok(ExpandLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Softmax" => Ok(SoftmaxLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Squeeze" => Ok(SqueezeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Unsqueeze" => Ok(UnsqueezeLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Erf" => Ok(ErfLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    "Conv" => Ok(ConvLayer::graph(&input_shapes, &node_constants, &node_attributes)),
    _ => Err(format!("Unsupported onnx operation: {op}")),
  }
  .unwrap()
}

// This function is used for updating the graph of onnx after creating a local graph
fn update_graph_w_local_graph(
  graph: &mut Graph,
  local_graph: Graph,
  node: &pb::NodeProto,
  input_idx: &HashMap<String, usize>,
  outputs_idx: &mut HashMap<String, Vec<(i32, usize)>>,
  basic_blocks_idx: &mut HashMap<String, usize>,
  models: &mut Vec<ArrayD<Fr>>,
) {
  // pushing basicblocks and models in a local graph to update graph.basic_blocks and models
  let mut local_block_idx = vec![];
  let temp = local_graph.basic_blocks;
  for basic_block in temp.into_iter() {
    let name = format!("{basic_block:?}");
    let idx = *basic_blocks_idx.entry(name).or_insert_with(|| graph.basic_blocks.len());
    local_block_idx.push(idx);
    if idx == graph.basic_blocks.len() {
      models.push(basic_block.genModel());
      graph.basic_blocks.push(basic_block);
    }
  }
  // pushing nodes in a local graph to update graph.nodes
  let start_idx = graph.nodes.len() as i32;
  for local_node in local_graph.nodes.iter() {
    // filter out node input that are ""
    let node_input = &node.input.iter().filter(|x| x as &str != "").collect::<Vec<_>>();
    graph.nodes.push(Node {
      basic_block: local_block_idx[local_node.basic_block],
      inputs: local_node
        .inputs
        .iter()
        .map(|(basicblock_idx, output_idx)| {
          if basicblock_idx < &0 {
            let input_tag = node_input[(-basicblock_idx - 1) as usize];
            if input_idx.contains_key(input_tag) {
              (-(*input_idx.get(input_tag).unwrap() as i32) - 1, *output_idx)
            } else {
              outputs_idx[input_tag][*output_idx]
            }
          } else {
            (start_idx + *basicblock_idx, *output_idx)
          }
        })
        .collect(),
    });
  }
  // tracking output_idx of local_graph
  for (i, output) in node.output.iter().enumerate() {
    let local_output = local_graph.outputs[i];
    outputs_idx.insert(output.clone(), vec![(start_idx + local_output.0, local_output.1)]);
  }
}

// This function is used for processing a layer of onnx models.
// It matches each onnx operation as a local graph. Then, it
// update the overall graph by adding the local graph.
fn process_node(
  node: &pb::NodeProto,
  graph: &mut Graph,
  shapes: &mut HashMap<String, Vec<usize>>,
  constants_hashmap: &HashMap<String, usize>,
  models: &mut Vec<ArrayD<Fr>>,
  passed_constants: &mut HashMap<String, ArrayD<Fr>>,
  input_idx: &HashMap<String, usize>,
  outputs_idx: &mut HashMap<String, Vec<(i32, usize)>>,
  basic_blocks_idx: &mut HashMap<String, usize>,
) {
  // match onnx operation
  let op = node.op_type.as_str();
  let input_shapes: Vec<_> = node.input.iter().map(|x| shapes.get(x)).collect();
  let input_shapes = input_shapes.into_iter().filter_map(|x| x).collect::<Vec<_>>(); // hack: we ignore optional inputs
  let node_constants = node.input.iter().map(|x| passed_constants.get(x).or(constants_hashmap.get(x).map(|&y| &models[y]))).collect();
  let node_attributes = node.attribute.iter().map(|x| x).collect();
  let (local_graph, output_shapes) = get_local_graph(op, &input_shapes, &node_constants, node_attributes);

  // compute precomputable constants (these are constants that can be computed without proving)
  if node_constants.iter().all(|&x| x.is_some()) {
    let node_inputs: Vec<_> = node_constants.iter().map(|&x| x.unwrap().clone()).collect();
    let node_inputs = node_inputs.iter().map(|x| x).collect();
    let outputs = local_graph.run(&node_inputs, &vec![&arr1(&[]).into_dyn(); local_graph.basic_blocks.len()]);
    node
      .output
      .iter()
      .zip(local_graph.outputs.iter())
      .zip(output_shapes.clone())
      .for_each(|((output_str, &(nodeX, nodeY)), output_shape)| {
        passed_constants.insert(
          output_str.to_string(),
          util::slice_nd_array(outputs[nodeX as usize][nodeY].clone(), &output_shape),
        );
      });
  }

  // update graph with local graph
  update_graph_w_local_graph(graph, local_graph, node, &input_idx, outputs_idx, basic_blocks_idx, models);

  // handle a special case (op == "Shape")
  if op == "Shape" {
    passed_constants.insert(
      (&node.output[0]).to_string(),
      arr1(&input_shapes[0].iter().map(|&x| Fr::from(x as i32)).collect::<Vec<_>>()).into_dyn(),
    );
  }

  // update shapes
  node.output.iter().zip(output_shapes).for_each(|(output, shape)| {
    shapes.insert(output.clone(), shape);
  });
}

// This function is used for loading onnx models and returning the graph and models
// - Graph: the graph of zk-torch BasicBlocks after parsing the onnx layers
// - Models: input tensors required for generating a setup for each BasicBlock
pub fn load_file(filename: &str) -> (Graph, Vec<ArrayD<Fr>>) {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let (input_idx, mut shapes) = parse_onnx_inputs(&onnx_graph);
  let (constants, constants_hashmap, mut models) = parse_onnx_constants(&onnx_graph, &mut shapes);

  let mut graph = Graph {
    basic_blocks: vec![],
    nodes: vec![],
    outputs: vec![],
  };
  let mut outputs_idx = create_output_indices(constants, &mut graph);

  let mut basic_blocks_idx = HashMap::new();
  let mut passed_constants = HashMap::new();

  for node in onnx_graph.node.iter().filter(|node| node.op_type.as_str() != "Constant") {
    process_node(
      node,
      &mut graph,
      &mut shapes,
      &constants_hashmap,
      &mut models,
      &mut passed_constants,
      &input_idx,
      &mut outputs_idx,
      &mut basic_blocks_idx,
    );
  }

  (graph, models)
}
