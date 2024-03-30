use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::util;
use ark_bn254::Fr;
use ndarray::{azip, ArrayD};
use std::collections::HashMap;
use std::error::Error;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, InferenceModelExt, SymbolValues, TypedFact, TypedOp};
use tract_onnx::tract_core::internal::DatumType;
use tract_onnx::tract_hir::ops::scan::Scan;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

pub struct match_node_notes {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
  pub models: Vec<ArrayD<f32>>,
  pub nonlinear_indices: Vec<usize>,
  const_tmp: Vec<ArrayD<f32>>,
  tract_node_to_const_id: HashMap<usize, usize>,
  tract_node_to_graph_node: HashMap<usize, usize>,
}

impl match_node_notes {
  pub fn new(basic_blocks: Vec<Box<dyn BasicBlock>>, nodes: Vec<Node>, models: Vec<ArrayD<f32>>) -> Self {
    let tract_node_to_const_id = HashMap::new();
    let tract_node_to_graph_node = HashMap::new();
    let const_tmp = vec![];
    let nonlinear_indices = vec![];
    Self {
      basic_blocks,
      nodes,
      models,
      nonlinear_indices,
      const_tmp,
      tract_node_to_const_id,
      tract_node_to_graph_node,
    }
  }
}

// load model weights
pub fn load_const_block(model: &Graph<TypedFact, Box<dyn TypedOp>>) -> HashMap<usize, ArrayD<f32>> {
  let mut tract_id_to_const_data = HashMap::new();
  for node in model.nodes.iter() {
    let node = match node.op().name().as_ref() {
      "Const" => {
        let idx = node.id;
        let op = load_op::<tract_onnx::tract_hir::ops::konst::Const>(node.op(), idx, node.op().name().to_string()).unwrap();
        // let dt = op.0.datum_type(); // Raw values are always f32
        let dims = op.0.shape().to_vec();
        let vals = op.0.as_slice::<f32>().unwrap().to_vec();
        let model = ArrayD::from_shape_vec(dims, vals).unwrap();
        tract_id_to_const_data.insert(idx, model);
      }
      &_ => {
        continue;
      }
    };
  }
  tract_id_to_const_data
}

pub fn match_node_op(model: Graph<TypedFact, Box<dyn TypedOp>>, notes: &mut match_node_notes) {
  for (idx, node) in model.nodes.iter().enumerate() {
    println!("Node {}: {:?}", idx, node);
    let node = match node.op().name().as_ref() {
      "Const" => {
        let op = load_op::<tract_onnx::tract_hir::ops::konst::Const>(node.op(), idx, node.op().name().to_string()).unwrap();
        // let dt = op.0.datum_type(); // Raw values are always f32
        let dims = op.0.shape().to_vec();
        let vals = op.0.as_slice::<f32>().unwrap().to_vec();
        let model = ArrayD::from_shape_vec(dims, vals).unwrap();
        notes.tract_node_to_const_id.insert(idx, notes.const_tmp.len());
        notes.const_tmp.push(model);
      }
      "Add" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::AddModelBasicBlock));
        let mut inputs = vec![];
        for (i, input_tract) in node.inputs.iter().enumerate() {
          let input_tract_node = input_tract.node;
          if notes.tract_node_to_const_id.contains_key(&input_tract_node) {
            let input_model_id = notes.tract_node_to_const_id.get(&input_tract_node).unwrap().clone();
            let model = notes.const_tmp[notes.tract_node_to_const_id.get(&input_tract_node).unwrap().clone()].clone();
            notes.models.push(model);
          } else if notes.tract_node_to_graph_node.contains_key(&input_tract_node) {
            let input_graph_node_id = notes.tract_node_to_graph_node.get(&input_tract_node).unwrap().clone();
            inputs.push((input_graph_node_id as i32, input_tract.slot));
          } else {
            panic!("Input node not found!");
          }
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_to_graph_node.insert(idx, notes.nodes.len() - 1);
      }
      "Mul" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::MulBasicBlock));
        let mut inputs = vec![];
        for (i, input_tract) in node.inputs.iter().enumerate() {
          let input_tract_node = input_tract.node;
          if notes.tract_node_to_const_id.contains_key(&input_tract_node) {
            let input_model_id = notes.tract_node_to_const_id.get(&input_tract_node).unwrap().clone();
            let model = notes.const_tmp[notes.tract_node_to_const_id.get(&input_tract_node).unwrap().clone()].clone();
            notes.models.push(model);
          } else if notes.tract_node_to_graph_node.contains_key(&input_tract_node) {
            let input_graph_node_id = notes.tract_node_to_graph_node.get(&input_tract_node).unwrap().clone();
            inputs.push((input_graph_node_id as i32, input_tract.slot));
          } else {
            panic!("Input node not found!");
          }
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_to_graph_node.insert(idx, notes.nodes.len() - 1);
      }
      "Max" => {
        if node.name.contains("Relu") {
          notes.basic_blocks.push(Box::new(crate::basic_block::ReLUBasicBlock));
          let input_node_id = notes.tract_node_to_graph_node.get(&node.inputs[0].node).unwrap().clone();
          notes.nonlinear_indices.push(notes.basic_blocks.len() - 1);
          notes.nodes.push(Node {
            basic_block: notes.basic_blocks.len() - 1,
            inputs: vec![(input_node_id as i32, 0)],
          });
          notes.models.push(ArrayD::zeros(vec![]));

          notes.basic_blocks.push(Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }));
          notes.nodes.push(Node {
            basic_block: notes.basic_blocks.len() - 1,
            inputs: vec![(input_node_id as i32, 0), (notes.nodes.len() as i32 - 1, 0)],
          });
          notes.models.push(ArrayD::zeros(vec![])); // should be cq table

          notes.tract_node_to_graph_node.insert(idx, notes.nodes.len()); // we insert two nodes
        } else {
          panic!("Max not implemented!");
        }
      }
      "Sigmoid" => {
        println!("Sigmoid to be implemented");
      }
      "EinSum" => {
        let op = load_op::<tract_onnx::tract_core::ops::einsum::EinSum>(node.op(), idx, node.op().name().to_string()).unwrap();
        let axes = &op.axes;
        match axes.to_string().as_ref() {
          "mk,nk->mn" => {
            // normal matrix multiplication for FeedForward layers
            println!("EinSum mk,nk->mn");
            notes.basic_blocks.push(Box::new(crate::basic_block::CQLinBasicBlock));

            if notes.tract_node_to_graph_node.keys().len() == 0 {
              notes.nodes.push(Node {
                basic_block: notes.basic_blocks.len() - 1,
                inputs: vec![(-1, 0)],
              });
            } else {
              let input_node_id = notes.tract_node_to_graph_node.get(&node.inputs[0].node).unwrap().clone();
              notes.nodes.push(Node {
                basic_block: notes.basic_blocks.len() - 1,
                inputs: vec![(input_node_id as i32, 0)],
              });
            }
            notes.tract_node_to_graph_node.insert(idx, notes.nodes.len() - 1);

            let model = notes.const_tmp[notes.tract_node_to_const_id.get(&node.inputs[1].node).unwrap().clone()].clone();
            notes.models.push(model.t().to_owned());
          }
          &_ => println!("Do nothing!"),
        }
      }
      &_ => println!("Do nothing!"),
    };
  }
}

pub fn create_updated_models(notes: &match_node_notes, scale_factor: u32) -> Vec<Vec<Vec<Fr>>> {
  let models = &notes.models;
  let mut updated_models = vec![];
  for model in models.iter() {
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
    updated_models.push(updated_model);
  }

  for nonlinear_idx in notes.nonlinear_indices.iter() {
    let (id, relu_cq_table) = util::gen_cq_table(vec![&notes.basic_blocks[*nonlinear_idx]], -(1 << 5), 1 << 6);
    let model = vec![id, relu_cq_table];
    updated_models[*nonlinear_idx+1] = model;
  }
  updated_models
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
        fact.shape.set_dim(i, tract_onnx::prelude::TDim::Val(1));
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
