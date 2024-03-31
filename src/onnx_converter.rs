use crate::basic_block::BasicBlock;
use crate::graph::Node;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::error::Error;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, InferenceModelExt, SymbolValues, TypedFact, TypedOp};
use tract_onnx::tract_core::internal::DatumType;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

pub struct MatchNodeNotes {
  pub basic_blocks: Vec<Box<dyn BasicBlock>>,
  pub nodes: Vec<Node>,
  pub tract_node_id: Vec<usize>,
}

impl MatchNodeNotes {
  pub fn new(basic_blocks: Vec<Box<dyn BasicBlock>>, nodes: Vec<Node>, tract_node_id: Vec<usize>) -> Self {
    Self {
      basic_blocks,
      nodes,
      tract_node_id,
    }
  }
}

// load Const block in tract graph as model weights
pub fn load_model_weights(model: &Graph<TypedFact, Box<dyn TypedOp>>) -> HashMap<usize, ArrayD<f32>> {
  let mut tract_id_to_const_data = HashMap::new();
  for node in model.nodes.iter() {
    let _ = match node.op().name().as_ref() {
      "Const" => {
        let idx = node.id;
        let op = load_op::<tract_onnx::tract_hir::ops::konst::Const>(node.op(), idx, node.op().name().to_string()).unwrap();
        // let dt = op.0.datum_type(); // Raw values are always f32
        let dims = op.0.shape().to_vec();
        let vals = op.0.as_slice::<f32>().unwrap().to_vec();
        let model = ArrayD::from_shape_vec(dims, vals).unwrap();
        tract_id_to_const_data.insert(idx, model);
      }
      &_ => continue,
    };
  }
  tract_id_to_const_data
}

// load tract graph and match nodes to basic blocks
pub fn load_tract_graph_basicblocks(model: Graph<TypedFact, Box<dyn TypedOp>>, scale_factor: u32) -> MatchNodeNotes {
  let basic_blocks = vec![];
  let nodes = vec![];
  let tract_node_id = vec![];
  let mut notes = MatchNodeNotes::new(basic_blocks, nodes, tract_node_id);
  for (idx, node) in model.nodes.iter().enumerate() {
    println!("Node {}: {:?}", idx, node);
    let _ = match node.op().name().as_ref() {
      "Add" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::AddModelBasicBlock));
        let mut inputs = vec![];
        for input_tract in node.inputs.iter() {
          inputs.push((input_tract.node as i32, input_tract.slot));
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_id.push(node.id);
      }
      "Div" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::DivScalarBasicBlock {
          output_SF: scale_factor as usize,
        }));
        let mut inputs = vec![];
        for input_tract in node.inputs.iter() {
          inputs.push((input_tract.node as i32, input_tract.slot));
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_id.push(node.id);
      }
      "EinSum" => {
        let op = load_op::<tract_onnx::tract_core::ops::einsum::EinSum>(node.op(), idx, node.op().name().to_string()).unwrap();
        let axes = &op.axes;
        let mut inputs = vec![];
        match axes.to_string().as_ref() {
          "mk,nk->mn" => { // Gemm
            // normal matrix multiplication for FeedForward layers
            notes.basic_blocks.push(Box::new(crate::basic_block::CQLinBasicBlock));

            for input_tract in node.inputs.iter() {
              inputs.push((input_tract.node as i32, input_tract.slot));
            }
            notes.nodes.push(Node {
              basic_block: notes.basic_blocks.len() - 1,
              inputs: inputs,
            });
            notes.tract_node_id.push(node.id);
          }
          "amk,kn->amn" => { // TODO: merge Lilia's CQLin matrix multiplication
            notes.basic_blocks.push(Box::new(crate::basic_block::CQLinBasicBlock));

            for input_tract in node.inputs.iter() {
              inputs.push((input_tract.node as i32, input_tract.slot));
            }
            notes.nodes.push(Node {
              basic_block: notes.basic_blocks.len() - 1,
              inputs: inputs,
            });
            notes.tract_node_id.push(node.id);
          }
          "amk,akn->amn" => { // TODO: merge Ari's MatMul after ndarray shape issues are resolved)
            notes.basic_blocks.push(Box::new(crate::basic_block::MatMulBasicBlock {l: 2})); // FIXME: hardcoded l=2 for now
            for input_tract in node.inputs.iter() {
              inputs.push((input_tract.node as i32, input_tract.slot));
            }
            notes.nodes.push(Node {
              basic_block: notes.basic_blocks.len() - 1,
              inputs: inputs,
            });
            notes.tract_node_id.push(node.id);

          }
          &_ => {
            panic!("Unsupported EinSum axes!");
          }
        }
      }
      "Max" => {
        if node.name.contains("Relu") {
          notes.basic_blocks.push(Box::new(crate::basic_block::ReLUBasicBlock));
          notes.nodes.push(Node {
            basic_block: notes.basic_blocks.len() - 1,
            inputs: vec![(node.inputs[0].node as i32, node.inputs[0].slot)],
          });
          notes.tract_node_id.push(node.id);

          notes.basic_blocks.push(Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }));
          notes.nodes.push(Node {
            basic_block: notes.basic_blocks.len() - 1,
            inputs: vec![
              (node.inputs[0].node as i32, node.inputs[0].slot),
              (node.inputs[0].node as i32, node.inputs[0].slot),
            ], // the second one should be output node
          });
          notes.tract_node_id.push(node.id);
        } else {
          panic!("Max not implemented!");
        }
      }
      "Mul" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::MulBasicBlock));
        let mut inputs = vec![];
        for input_tract in node.inputs.iter() {
          inputs.push((input_tract.node as i32, input_tract.slot));
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_id.push(node.id);
      }
      "Sigmoid" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::SigmoidBasicBlock {
          input_SF: scale_factor as usize,
          output_SF: scale_factor as usize,
        }));
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: vec![(node.inputs[0].node as i32, node.inputs[0].slot)],
        });
        notes.tract_node_id.push(node.id);

        notes.basic_blocks.push(Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }));
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: vec![
            (node.inputs[0].node as i32, node.inputs[0].slot),
            (node.inputs[0].node as i32, node.inputs[0].slot),
          ], // the second one should be output node
        });
        notes.tract_node_id.push(node.id);
      }
      "Sqrt" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::SqrtBasicBlock {
          input_SF: scale_factor as usize,
          output_SF: scale_factor as usize,
        }));
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: vec![(node.inputs[0].node as i32, node.inputs[0].slot)],
        });
        notes.tract_node_id.push(node.id);

        notes.basic_blocks.push(Box::new(crate::basic_block::CQ2BasicBlock { table_dict: HashMap::new() }));
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: vec![
            (node.inputs[0].node as i32, node.inputs[0].slot),
            (node.inputs[0].node as i32, node.inputs[0].slot),
          ], // the second one should be output node
        });
        notes.tract_node_id.push(node.id);
      }
      "Transpose" => {
        notes.basic_blocks.push(Box::new(crate::basic_block::TransposeBasicBlock));
        let mut inputs = vec![];
        for input_tract in node.inputs.iter() {
          inputs.push((input_tract.node as i32, input_tract.slot));
        }
        notes.nodes.push(Node {
          basic_block: notes.basic_blocks.len() - 1,
          inputs: inputs,
        });
        notes.tract_node_id.push(node.id);
      }
      &_ => continue, // skip unsupported ops for now
    };
  }
  notes
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

// convert tract graph to zkllm graph (reindexing inputs and creating models)
pub fn convert_tract_to_zkg(
  tract_graph_basicblocks: &mut MatchNodeNotes,
  model_weight: HashMap<usize, ArrayD<f32>>,
  scale_factor: u32,
) -> (Vec<Node>, Vec<Vec<Vec<Fr>>>) {
  let mut tract_to_zkllm_id: HashMap<usize, i32> = HashMap::new();
  let mut models = vec![];
  let mut inputs = vec![];
  let mut no_model = true;
  for (i, n) in tract_graph_basicblocks.nodes.iter().enumerate() {
    // print basicblock name, self id, and inputs
    // println!("{i} {:?}", tract_graph_basicblocks.basic_blocks[n.basic_block].name());
    // println!("{i} {:?}", tract_graph_basicblocks.tract_node_id[i]);
    // println!("{i} {:?}", n.inputs);

    let name = tract_graph_basicblocks.basic_blocks[n.basic_block].name();

    // convert inputs
    let mut updated_inputs = vec![];
    for (j, input) in n.inputs.iter().enumerate() {
      let input_node = input.0 as usize;
      let input_slot = input.1 as usize;
      if model_weight.contains_key(&input_node) {
        let model = model_weight.get(&input_node).unwrap();
        let model = convert_array_to_vec(model.clone(), scale_factor);
        models.push(model);
        no_model = false;
      } else if name == "CQ2" && j == 1 {
        let (id, relu_cq_table) = util::gen_cq_table(vec![&tract_graph_basicblocks.basic_blocks[i - 1]], -(1 << 5), 1 << 6);
        let model = vec![id, relu_cq_table];
        models.push(model);
        updated_inputs.push(((i - 1) as i32, input_slot));
        no_model = false;
      } else {
        if input_node == 0 {
          // input node
          updated_inputs.push((-1 as i32, input_slot));
        } else if tract_to_zkllm_id.contains_key(&input_node) {
          // internal node
          let input_id = tract_to_zkllm_id.get(&input_node).unwrap();
          updated_inputs.push((*input_id, input_slot));
        } else {
          panic!("Input node not found!");
        }
      }
    }
    if name != "CQ2" {
      tract_to_zkllm_id.insert(tract_graph_basicblocks.tract_node_id[i], i as i32);
    }
    if no_model {
      models.push(vec![]);
    } else {
      no_model = true;
    }
    inputs.push(Node {
      basic_block: n.basic_block,
      inputs: updated_inputs,
    });
  }
  (inputs, models)
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
