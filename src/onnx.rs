use crate::basic_block::*;
use crate::graph::*;
use crate::layer::*;
use crate::util;
use crate::CONFIG;
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{arr1, ArrayD};
use once_cell::sync::Lazy;
use pool::MaxPoolLayer;
use std::collections::HashMap;
use tract_onnx::pb;
use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::Datum;
use tract_onnx::prelude::{DatumType, Framework};
use tract_onnx::tensor::load_tensor;

pub static SF_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.scale_factor_log);
pub static SF: Lazy<usize> = Lazy::new(|| 1 << *SF_LOG);
pub static SF_FLOAT: Lazy<f32> = Lazy::new(|| (1 << *SF_LOG) as f32);
pub static CQ_RANGE: Lazy<usize> = Lazy::new(|| 1 << CONFIG.sf.cq_range_log);
pub static CQ_RANGE_LOWER: Lazy<i32> = Lazy::new(|| -(1 << CONFIG.sf.cq_range_lower_log));

// This function is used for parsing the inputs of onnx models
fn parse_onnx_inputs(onnx_graph: &pb::GraphProto) -> (HashMap<String, usize>, HashMap<String, Vec<usize>>, HashMap<String, DatumType>) {
  let mut input_idx = HashMap::new();
  let mut shapes = HashMap::new();
  let mut types = HashMap::new();

  for (idx, i) in onnx_graph.input.iter().enumerate() {
    input_idx.insert(i.name.clone(), idx);
    let tract_onnx::pb::type_proto::Value::TensorType(t) = i.r#type.as_ref().unwrap().value.as_ref().unwrap();
    shapes.insert(
      i.name.clone(),
        util::get_shape_from_onnx_tensor(t)
    );
    types.insert(i.name.clone(), util::datatype_to_datumtype(t.elem_type));
  }

  (input_idx, shapes, types)
}

// This function is used for parsing the constants of onnx models
fn parse_onnx_constants<'a>(
  onnx_graph: &'a pb::GraphProto,
  shapes: &mut HashMap<String, Vec<usize>>,
  types: &mut HashMap<String, DatumType>,
) -> (
  impl Iterator<Item = (String, &'a pb::TensorProto)> + 'a,
  HashMap<String, usize>,
  Vec<(ArrayD<Fr>, DatumType)>,
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
    let (tensor, data_type) = if tensor.data_location.is_none() {
      let tensor = load_tensor(&*onnx.provider, tensor, None).unwrap();
      let data_type = tensor.datum_type();
      let tensor = match tensor.datum_type() {
        DatumType::F32 => {
          let tensor = tensor.into_array::<f32>().unwrap();
          Ok(tensor.map(|x| {
            // handle the case where the constant is very close to zero (i.e., epsilon to prevent division by zero)
            if *x < 1e-10 && *x > 0.0 {
              // the reason we use 1 here is because it is the smallest positive value that can be represented in the field
              return Fr::from(1);
            }
            let mut y = (*x * *SF_FLOAT).round();
            y = y.clamp(-(1 << 15) as f32, (1 << 15) as f32);
            Fr::from(y as i32)
          }))
        }
        DatumType::I64 => {
          let tensor = tensor.into_array::<i64>().unwrap();
          Ok(tensor.map(|x| Fr::from(*x)))
        }
        DatumType::Bool => {
          let tensor = tensor.into_array::<bool>().unwrap();
          Ok(tensor.map(|x| Fr::from(*x as i32)))
        }
        _ => Err(format!("Unsupported constant type: {:?}", tensor.datum_type())),
      }
      .unwrap();
      (tensor, data_type)
    } else {
      // if the data_location is not None, we generate fake weights for now.
      // TODO: we can add support for loading weights from file later
      let shape: Vec<usize> = tensor.dims.iter().map(|&i| i as usize).collect();
      let dtype = DataType::from_i32(tensor.data_type).unwrap().into();
      let data_type = util::datatype_to_datumtype(tensor.data_type);
      let tensor = util::generate_fake_tensor(dtype, shape);
      (tensor, data_type)
    };

    shapes.insert(name.clone(), tensor.shape().to_vec());
    types.insert(name.clone(), data_type);
    let tensor = util::pad_to_pow_of_two(&tensor, &Fr::zero());
    constants_hashmap.insert(name.clone(), idx);
    models.push((tensor, data_type));
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
    graph.layer_names.push(format!("Const {}", name.to_string()));
    graph.basic_blocks.push(Box::new(ConstBasicBlock {}));
    graph.precomputable.setup.push(false);
    graph.precomputable.prove_and_verify.push(true);
    graph.precomputable.encodeOutputs.push(true);
  }

  outputs_idx
}

// This function is used for getting the local graph for each layer
fn get_local_graph(
  op: &str,
  input_shapes: &Vec<&Vec<usize>>,
  input_types: &Vec<DatumType>,
  node_constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
  node_attributes: Vec<&AttributeProto>,
) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
  match op {
    "Add" => Ok(AddLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "And" => Ok(AndLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "ArgMax" => Ok(ArgMaxLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Mul" => Ok(MulLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Cast" => Ok(CastLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Identity" => Ok(CastLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)), // Identity is equivalent to Cast in zk-torch
    "InstanceNormalization" => Ok(InstanceNormLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "BatchNormalization" => Ok(BatchNormLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Ceil" => Ok(CeilLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Clip" => Ok(ClipLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Concat" => Ok(ConcatLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "ConstantOfShape" => Ok(ConstOfShapeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Cos" => Ok(CosLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Sin" => Ok(SinLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Sub" => Ok(SubLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Einsum" => Ok(EinsumLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Less" => Ok(LessLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "LSTM" => Ok(LSTMLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "MatMul" => Ok(MatMulLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Mod" => Ok(ModLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Neg" => Ok(NegLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Not" => Ok(NotLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Relu" => Ok(ReLULayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Flatten" => Ok(FlattenLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Gather" => Ok(GatherLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "GatherND" => Ok(GatherNDLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Gemm" => Ok(GemmLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Range" => Ok(RangeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Reciprocal" => Ok(ReciprocalLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "ReduceMean" => Ok(ReduceMeanLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Pow" => Ok(PowLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Div" => Ok(DivLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "ScatterND" => Ok(ScatterNDLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Slice" => Ok(SliceLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Split" => Ok(SplitLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Sqrt" => Ok(SqrtLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Reshape" => Ok(ReshapeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Resize" => Ok(ResizeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Transpose" => Ok(TransposeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Tan" => Ok(TanLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Tanh" => Ok(TanhLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "TopK" => Ok(TopKLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Tile" => Ok(TileLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Shape" => Ok(ShapeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Sigmoid" => Ok(SigmoidLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Equal" => Ok(EqualLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Where" => Ok(WhereLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Expand" => Ok(ExpandLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Softmax" => Ok(SoftmaxLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Squeeze" => Ok(SqueezeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Unsqueeze" => Ok(UnsqueezeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Erf" => Ok(ErfLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Conv" => Ok(ConvLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "ConvTranspose" => Ok(ConvTransposeLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Max" => Ok(MaxLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Min" => Ok(MinLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "MaxPool" => Ok(MaxPoolLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
    "Xor" => Ok(XorLayer::graph(&input_shapes, &input_types, &node_constants, &node_attributes)),
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
  models: &mut Vec<(ArrayD<Fr>, DatumType)>,
  precomputable: bool,
) {
  // pushing basicblocks and models in a local graph to update graph.basic_blocks and models
  let mut local_block_idx = vec![];
  let temp = local_graph.basic_blocks;
  for basic_block in temp.into_iter() {
    let name = format!("{basic_block:?}");
    let idx = *basic_blocks_idx.entry(name).or_insert_with(|| graph.basic_blocks.len());
    local_block_idx.push(idx);
    if idx == graph.basic_blocks.len() {
      models.push((basic_block.genModel(), DatumType::I64));
      graph.basic_blocks.push(basic_block);
      graph.precomputable.setup.push(precomputable);
    } else {
      graph.precomputable.setup[idx] = graph.precomputable.setup[idx] && precomputable;
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
    let name = if node.name.clone() == "" {
      node.op_type.to_string()
    } else {
      node.name.clone()
    };
    if precomputable {
      graph.layer_names.push(format!("Op {} (precomputed)", name));
    } else {
      graph.layer_names.push(format!("Op {}", name));
    }
    graph.precomputable.prove_and_verify.push(precomputable);
    graph.precomputable.encodeOutputs.push(precomputable);
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
  types: &mut HashMap<String, DatumType>,
  constants_hashmap: &HashMap<String, usize>,
  models: &mut Vec<(ArrayD<Fr>, DatumType)>,
  passed_constants: &mut HashMap<String, ArrayD<Fr>>,
  input_idx: &HashMap<String, usize>,
  outputs_idx: &mut HashMap<String, Vec<(i32, usize)>>,
  basic_blocks_idx: &mut HashMap<String, usize>,
) {
  let mut precomputable = false;
  // match onnx operation
  let op = node.op_type.as_str();
  println!("Compiling ONNX node: {}", node.name);
  let input_shapes: Vec<_> = node.input.iter().map(|x| shapes.get(x)).collect();
  let input_shapes = input_shapes.into_iter().filter_map(|x| x).collect::<Vec<_>>(); // hack: we ignore optional inputs
  let input_types: Vec<_> = node.input.iter().map(|x| types.get(x)).collect();
  let input_types = input_types.into_iter().filter_map(|opt| opt.map(|x| *x)).collect::<Vec<_>>();
  let node_constants = node
    .input
    .iter()
    .map(|x| {
      if let Some(a) = passed_constants.get(x) {
        Some((a, DatumType::I64))
      } else {
        constants_hashmap.get(x).map(|&y| (&models[y].0, models[y].1))
      }
    })
    .collect();
  let node_attributes = node.attribute.iter().map(|x| x).collect();
  let (local_graph, output_shapes, output_types) = get_local_graph(op, &input_shapes, &input_types, &node_constants, node_attributes);

  // compute precomputable constants (these are constants that can be computed without proving)
  if node_constants.iter().all(|&x| x.is_some()) {
    let node_inputs: Vec<_> = node_constants.iter().map(|&x| x.unwrap().clone()).collect();
    let node_inputs = node_inputs.iter().map(|x| x.0).collect();
    let outputs = local_graph.run(&node_inputs, &vec![&arr1(&[]).into_dyn(); local_graph.basic_blocks.len()]);
    node.output.iter().zip(local_graph.outputs.iter()).for_each(|(output_str, &(nodeX, nodeY))| {
      passed_constants.insert(output_str.to_string(), outputs[nodeX as usize][nodeY].clone());
    });
    precomputable = true;
  }

  // update graph with local graph
  update_graph_w_local_graph(graph, local_graph, node, &input_idx, outputs_idx, basic_blocks_idx, models, precomputable);

  // handle a special case (op == "Shape")
  if op == "Shape" {
    let shape = arr1(&input_shapes[0].iter().map(|&x| Fr::from(x as i32)).collect::<Vec<_>>()).into_dyn();
    passed_constants.insert((&node.output[0]).to_string(), util::pad_to_pow_of_two(&shape, &Fr::zero()));
  }

  // update shapes
  node.output.iter().zip(output_shapes).zip(output_types).for_each(|((output, shape), t)| {
    shapes.insert(output.clone(), shape);
    types.insert(output.clone(), t);
  });
}

// This function is used for finding all the skip-able nodes when encoding the circuit outputs.
// The high-level idea is that we can skip encodeOutputs for a node only when
// - the node itself is precomputable
// - all of its outputs are fed into precomputable nodes (that's why we need to propagate precomputable)
fn propagate_precomputable(graph: &mut Graph) {
  println!("Determining skip-able nodes when encoding the circuit...");
  let mut precomputable = graph.precomputable.encodeOutputs.clone();
  let mut changed = true;
  let mut counter = 0;
  // stop propagating when no more changes are made
  while changed {
    println!("  Iteration: {}", counter);
    changed = false;
    for i in 0..graph.nodes.len() {
      let node = &graph.nodes[i];
      for inp in node.inputs.iter() {
        if inp.0 >= 0 {
          let orig = precomputable[inp.0 as usize];
          precomputable[inp.0 as usize] = precomputable[inp.0 as usize] && precomputable[i];
          changed = changed || (orig != precomputable[inp.0 as usize]);
        }
      }
    }
    counter += 1;
  }
  graph.precomputable.encodeOutputs = precomputable;
}

// This function is used for loading onnx models and returning the graph and models
// - Graph: the graph of zk-torch BasicBlocks after parsing the onnx layers
// - Models: input tensors required for generating a setup for each BasicBlock
pub fn load_file(filename: &str) -> (Graph, Vec<(ArrayD<Fr>, DatumType)>) {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let (input_idx, mut shapes, mut types) = parse_onnx_inputs(&onnx_graph);
  let (constants, constants_hashmap, mut models) = parse_onnx_constants(&onnx_graph, &mut shapes, &mut types);

  let mut graph = Graph {
    basic_blocks: vec![],
    precomputable: Precomputable::new(),
    layer_names: vec![],
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
      &mut types,
      &constants_hashmap,
      &mut models,
      &mut passed_constants,
      &input_idx,
      &mut outputs_idx,
      &mut basic_blocks_idx,
    );
  }

  propagate_precomputable(&mut graph);

  (graph, models)
}
