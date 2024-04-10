use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::util;
use ark_bn254::Fr;
use itertools::izip;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::error::Error;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, InferenceModelExt, Node as OnnxNode, SymbolValues, TypedFact, TypedOp};
use tract_onnx::tract_core::internal::DatumType;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

// load Const block in tract graph, node names, and output shapes
pub fn load_tract_layers(model: &Graph<TypedFact, Box<dyn TypedOp>>, symbol_values: &SymbolValues) -> HashMap<usize, ArrayD<f32>> {
  let mut weights_map = HashMap::new();
  let mut node_names = vec![];
  let mut output_shapes = vec![];
  let mut inputs = vec![];
  for node in model.nodes.iter() {
    println!("Node: {:?}", node);
    let name = String::from(node.op().name().as_ref());
    node_names.push(name.clone());
    let output_shape = node_output_shapes(&node, symbol_values).unwrap();
    output_shapes.push(output_shape);
    let mut node_inputs = vec![];
    for input_tract in node.inputs.iter() {
      node_inputs.push((input_tract.node as i32, input_tract.slot));
    }
    inputs.push(node_inputs);
    let _ = match &*name {
      "Const" => {
        let idx = node.id;
        let op = load_op::<tract_onnx::tract_hir::ops::konst::Const>(node.op(), idx, node.op().name().to_string()).unwrap();
        // let dt = op.0.datum_type(); // Raw values are always f32
        let dims = op.0.shape().to_vec();
        let vals = op.0.as_slice::<f32>().unwrap().to_vec();
        let model = ArrayD::from_shape_vec(dims, vals).unwrap();
        weights_map.insert(idx, model);
      }
      &_ => continue,
    };
  }
  weights_map
}

// load tract layers and match them to basic blocks
pub fn convert_tract_to_basicblocks(
  weights_map: HashMap<usize, ArrayD<f32>>,
  tract_model: &Graph<TypedFact, Box<dyn TypedOp>>,
  scale_factor: u32,
) -> (Vec<Box<dyn BasicBlock>>, Vec<Node>, Vec<Vec<Vec<Fr>>>) {
  let mut basic_blocks: Vec<Box<dyn BasicBlock>> = Vec::new();
  let mut nodes: Vec<Node> = Vec::new();
  let mut model_weights: Vec<Vec<Vec<Fr>>> = Vec::new();
  let mut tract_node_id_to_zkllm_node_id = HashMap::new();
  let mut start_node_id = 0;
  let mut no_model_update = true;
  for (idx, node) in tract_model.nodes.iter().enumerate() {
    println!("Node: {:?}", node);
    let _ = match node.op().name().as_ref() {
      "Abs" => {
        let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
          Box::new(crate::basic_block::AbsBasicBlock),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
        ];
        let mut new_nodes = vec![
          Node {
            basic_block: basic_blocks.len(),
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 1,
            inputs: vec![(-1, 0), (0, 0)], // the second one should be output node
          },
        ];
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
        basic_blocks.append(&mut new_blocks);
        nodes.append(&mut new_nodes);
      }
      "Add" => {
        // currently only support Add for input+model
        basic_blocks.push(Box::new(crate::basic_block::AddModelBasicBlock));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0)], // only one input, another input is model
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() - 1);
      }
      "Div" => { // TO add CQ2
        basic_blocks.push(Box::new(crate::basic_block::DivScalarBasicBlock {
          output_SF: scale_factor as usize,
        }));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0), (-1, 1)], // first input is dividend, second input is divisor
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() - 1);
      }
      "EinSum" => {
        let op = load_op::<tract_onnx::tract_core::ops::einsum::EinSum>(node.op(), idx, node.op().name().to_string()).unwrap();
        let axes = &op.axes;
        match axes.to_string().as_ref() {
          "mk,nk->mn" => {
            // Gemm
            // normal matrix multiplication for FeedForward layers
            basic_blocks.push(Box::new(crate::basic_block::CQLinBasicBlock));
            nodes.push(Node {
              basic_block: basic_blocks.len() - 1,
              inputs: vec![(-1, 0)],
            });
            tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() - 1);
          }
          "amk,kn->amn" => {
            basic_blocks.push(Box::new(crate::basic_block::MatMulFixedBasicBlock));
            nodes.push(Node {
              basic_block: basic_blocks.len() - 1,
              inputs: vec![(-1, 0)],
            });
            tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() - 1);
          }
          "amk,akn->amn" => {
            // TODO: merge Ari's MatMul after ndarray shape issues are resolved)
            panic!("MatMul not implemented!");
          }
          &_ => {
            panic!("Unsupported EinSum axes!");
          }
        }
      }
      "Max" => {
        if node.name.contains("Relu") {
          let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
            Box::new(crate::basic_block::ReLUBasicBlock),
            Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
          ];
          let mut new_nodes = vec![
            Node {
              basic_block: basic_blocks.len(),
              inputs: vec![(-1, 0)],
            },
            Node {
              basic_block: basic_blocks.len() + 1,
              inputs: vec![(-1, 0), (0, 0)], // the second one should be output node
            },
          ];
          tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
          basic_blocks.append(&mut new_blocks);
          nodes.append(&mut new_nodes);
        } else {
          panic!("Max not implemented!");
        }
      }
      "Mul" => {
        basic_blocks.push(Box::new(crate::basic_block::MulBasicBlock));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0), (-1, 1)],
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
      }
      "Recip" => {
        let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
          Box::new(crate::basic_block::ReciprocalBasicBlock),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
        ];
        let mut new_nodes = vec![
          Node {
            basic_block: basic_blocks.len(),
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 1,
            inputs: vec![(-1, 0), (0, 0)], // the second one should be output node
          },
        ];
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
        basic_blocks.append(&mut new_blocks);
        nodes.append(&mut new_nodes);
      }
      "Reduce<Sum>" => {
        let op = load_op::<tract_onnx::tract_core::ops::nn::Reduce>(node.op(), idx, node.op().name().to_string()).unwrap();
        let axes: Vec<_> = op.axes.into_iter().collect();
        basic_blocks.push(Box::new(crate::basic_block::SumBasicBlock));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0)],
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
      }
      "Sigmoid" => {
        let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
          Box::new(crate::basic_block::SigmoidBasicBlock {
            input_SF: scale_factor as usize,
            output_SF: scale_factor as usize,
          }),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
        ];
        let mut new_nodes = vec![
          Node {
            basic_block: basic_blocks.len(),
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 1,
            inputs: vec![(-1, 0), (0, 0)], // the second one should be output node
          },
        ];
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
        basic_blocks.append(&mut new_blocks);
        nodes.append(&mut new_nodes);
      }
      "Sqrt" => {
        let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
          Box::new(crate::basic_block::SqrtBasicBlock {
            input_SF: scale_factor as usize,
            output_SF: scale_factor as usize,
          }),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
        ];
        let mut new_nodes = vec![
          Node {
            basic_block: basic_blocks.len(),
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 1,
            inputs: vec![(-1, 0), (0, 0)], // the second one should be output node
          },
        ];
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
        basic_blocks.append(&mut new_blocks);
        nodes.append(&mut new_nodes);
      }
      "Square" => {
        basic_blocks.push(Box::new(crate::basic_block::MulBasicBlock));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0), (-1, 0)],
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len());
      }
      "Softmax" => {
        let mut new_blocks: Vec<Box<dyn BasicBlock>> = vec![
          Box::new(crate::basic_block::MaxBasicBlock),
          Box::new(crate::basic_block::SubBasicBlock),
          Box::new(crate::basic_block::ExpBasicBlock {
            input_SF: scale_factor as usize,
            output_SF: scale_factor as usize,
          }),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
          Box::new(crate::basic_block::SumBasicBlock),
          Box::new(crate::basic_block::LogBasicBlock {
            input_SF: scale_factor as usize,
            output_SF: scale_factor as usize,
          }),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
          Box::new(crate::basic_block::AddBasicBlock),
          Box::new(crate::basic_block::SubBasicBlock),
          Box::new(crate::basic_block::ExpBasicBlock {
            input_SF: scale_factor as usize,
            output_SF: scale_factor as usize,
          }),
          Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }),
        ];
        let mut new_nodes = vec![
          Node {
            basic_block: basic_blocks.len(),
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 1,
            inputs: vec![(-1, 0), (0, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 2,
            inputs: vec![(1, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 3,
            inputs: vec![(1, 0), (2, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 4,
            inputs: vec![(2, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 5,
            inputs: vec![(4, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 6,
            inputs: vec![(4, 0), (5, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 7,
            inputs: vec![(0, 0), (5, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 8,
            inputs: vec![(-1, 0), (7, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 9,
            inputs: vec![(8, 0)],
          },
          Node {
            basic_block: basic_blocks.len() + 10,
            inputs: vec![(8, 0), (9, 0)],
          },
        ];
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() + 9);
        basic_blocks.append(&mut new_blocks);
        nodes.append(&mut new_nodes);
      }
      "Transpose" => {
        basic_blocks.push(Box::new(crate::basic_block::TransposeBasicBlock));
        nodes.push(Node {
          basic_block: basic_blocks.len() - 1,
          inputs: vec![(-1, 0)],
        });
        tract_node_id_to_zkllm_node_id.insert(node.id, nodes.len() - 1);
      }
      &_ => continue, // skip unsupported ops for now
    };

    // shift input indices for the next model
    for i in start_node_id..nodes.len() {
      for (j, input) in &mut nodes[i].inputs.iter_mut().enumerate() {
        if input.0 >= 0 {
          *input = (input.0 + start_node_id as i32, input.1);
          if basic_blocks[i].name() == "CQ2" && no_model_update {
            let (id, relu_cq_table) = util::gen_cq_table(vec![&basic_blocks[i - 1]], -(1 << 5), 1 << 6);
            let model_weight = vec![id, relu_cq_table];
            model_weights.push(model_weight);
            no_model_update = false;
          }
        } else {
          // -1 means external input
          let tract_input = node.inputs[input.1];
          for tract_node_input in node.inputs.iter() {
            if weights_map.contains_key(&tract_node_input.node) && basic_blocks[i].name() != "CQ2" {
              let model_weight = weights_map.get(&tract_node_input.node).unwrap();
              let model_weight = convert_array_to_vec(model_weight.clone(), scale_factor);
              model_weights.push(model_weight);
              no_model_update = false;
            }
          }
          if tract_node_id_to_zkllm_node_id.contains_key(&tract_input.node) {
            *input = (tract_node_id_to_zkllm_node_id[&tract_input.node].try_into().unwrap(), tract_input.slot);
          } else if tract_model.nodes[tract_input.node].op().name().as_ref() == "Source" {
            *input = (-1, tract_input.slot);
          } else {
            panic!("Input node not found in tract_node_id_to_zkllm_node_id!");
          }
        }
      }

      if no_model_update {
        model_weights.push(vec![]);
      } else {
        no_model_update = true;
      }
    }
    start_node_id = nodes.len();
  }

  (basic_blocks, nodes, model_weights)
}

// temporary function to convert ArrayD<f32> to Vec<Vec<Fr>> (we don't need this after we adopt ArrayD<Fr> in Basicblocks)
pub fn convert_array_to_vec(model: ArrayD<f32>, scale_factor: u32) -> Vec<Vec<Fr>> {
  let mut updated_model = vec![];
  if model.ndim() == 0 {
    updated_model.push(vec![]);
  } else if model.ndim() == 1 {
    let mut updated_model_d = vec![];
    for val in model.iter() {
      updated_model_d.push(Fr::from((val.clone() * scale_factor as f32).round() as i32));
    }
    updated_model.push(updated_model_d);
  } else if model.ndim() == 2 {
    let model_shape = model.shape();
    for i in 0..model_shape[0] {
      let mut updated_model_d = vec![];
      for j in 0..model_shape[1] {
        updated_model_d.push(Fr::from((model[[i, j]] * scale_factor as f32).round() as i32));
      }
      updated_model.push(updated_model_d);
    }
  } else {
    panic!("Unsupported model shape!");
  }
  updated_model
}

fn load_op<C: tract_onnx::prelude::Op + Clone>(op: &dyn tract_onnx::prelude::Op, idx: usize, name: String) -> Result<C, Box<dyn std::error::Error>> {
  let op: &C = match op.downcast_ref::<C>() {
    Some(b) => b,
    None => {
      return Err(format!("Expected {} at index {} to be a {}", name, idx, std::any::type_name::<C>()).into());
    }
  };

  Ok(op.clone())
}

pub fn load_onnx_tract_model(path: &str) -> Result<TractResult, Box<dyn Error>> {
  use tract_onnx::{tract_core::internal::IntoArcTensor, tract_hir::internal::GenericFactoid};

  let mut reader = std::fs::File::open(path).unwrap();
  let mut model = tract_onnx::onnx().model_for_read(&mut reader).unwrap();

  for (i, id) in model.clone().inputs.iter().enumerate() {
    let input = model.node_mut(id.node);
    let mut fact: InferenceFact = input.outputs[0].fact.clone();

    for (i, x) in fact.clone().shape.dims().enumerate() {
      if matches!(x, GenericFactoid::Any) {
        let batch_size = 1;
        fact.shape.set_dim(i, tract_onnx::prelude::TDim::Val(batch_size as i64));
      }
    }

    model.set_input_fact(i, fact)?;
  }

  for (i, _) in model.clone().outputs.iter().enumerate() {
    model.set_output_fact(i, InferenceFact::default())?;
  }

  let symbol_values = SymbolValues::default();

  // Note: do not optimize the model, as the layout will depend on underlying hardware
  let mut typed_model = model.into_typed()?.concretize_dims(&symbol_values)?.into_decluttered()?;

  // concretize constants
  for node in typed_model.eval_order()? {
    let node = typed_model.node_mut(node);
    if let Some(op) = node.op_as_mut::<tract_onnx::tract_core::ops::konst::Const>() {
      if op.0.datum_type() == DatumType::TDim {
        // get inner value to Arc<Tensor>
        let mut constant = op.0.as_ref().clone();
        // Generally a shape or hyperparam
        constant.as_slice_mut::<tract_onnx::prelude::TDim>()?.iter_mut().for_each(|x| *x = x.eval(&symbol_values));

        op.0 = constant.into_arc_tensor();
      }
    }
  }

  Ok((typed_model, symbol_values))
}

pub fn node_output_shapes(
  node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
  symbol_values: &SymbolValues,
) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>> {
  let mut shapes = Vec::new();
  let outputs = node.outputs.to_vec();
  for output in outputs {
    let mut result = vec![];
    let shape = output.fact.shape;
    let mv = shape.to_vec();
    for (i, x) in mv.iter().enumerate() {
      // case 1: Val(x)
      if let tract_onnx::prelude::TDim::Val(x) = x {
        result.push(*x as i32);
      }
      // case 2: Sym(batch_size)
      else if let tract_onnx::prelude::TDim::Sym(x) = x {
        result.push(-1);
      }
    }
    shapes.push(result)
  }
  Ok(shapes)
}
