use crate::basic_block::*;
use crate::graph::*;
use ark_bn254::Fr;
use ndarray::ArrayD;
use std::collections::HashMap;
use tract_onnx::prelude::Framework;
use tract_onnx::tensor::load_tensor;

pub fn load_file(filename: &str) -> (Graph, Vec<ArrayD<Fr>>) {
  let onnx = tract_onnx::onnx();
  let onnx_graph = onnx.proto_model_for_path(filename).unwrap().graph.unwrap();

  let mut graph = Graph {
    basic_blocks: vec![],
    nodes: vec![],
    outputs: vec![],
  };
  let mut basic_blocks_idx: HashMap<String, usize> = HashMap::new(); //BasicBlock to graph.basic_blocks index
  let mut outputs_idx: HashMap<String, Vec<(i32, usize)>> = HashMap::new(); //Graph node name to graph.nodes outputs
  let mut setups = vec![];

  for tensor in onnx_graph.initializer {
    let name = tensor.name.clone();
    let tensor = load_tensor(&*onnx.provider, &tensor, None).unwrap();
    let tensor = tensor.into_array::<f32>().unwrap();
    let tensor = tensor.map(|x| Fr::from((*x * ((1 << 3) as f32)).round() as i32));
    outputs_idx.insert(name, vec![(graph.basic_blocks.len() as i32, 0)]);
    graph.nodes.push(Node {
      basic_block: graph.basic_blocks.len(),
      inputs: vec![],
    });
    graph.basic_blocks.push(Box::new(ConstBasicBlock {}));
    setups.push(tensor);
  }
  for node in onnx_graph.node {
    let op = node.op_type.as_str();
    let mut local_graph = match op {
      "Add" => Ok(Graph {
        basic_blocks: vec![Box::new(AddBasicBlock {})],
        nodes: vec![Node {
          basic_block: 0,
          inputs: vec![(-1, 0), (-2, 0)],
        }],
        outputs: vec![(0, 0)],
      }),
      "MatMul" => Ok(Graph {
        basic_blocks: vec![Box::new(MatMulBasicBlock {})],
        nodes: vec![Node {
          basic_block: 0,
          inputs: vec![(-1, 0), (-2, 0)],
        }],
        outputs: vec![(0, 0)],
      }),
      "Relu" => Ok(Graph {
        basic_blocks: vec![
          Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }),
          Box::new(CQ2BasicBlock {
            table_dict: HashMap::new(),
            setup: Some((Box::new(ReLUBasicBlock { input_SF: 3, output_SF: 3 }), -(1 << 3), 1 << 4)),
          }),
        ],
        nodes: vec![
          Node {
            basic_block: 0,
            inputs: vec![(-1, 0)],
          },
          Node {
            basic_block: 1,
            inputs: vec![(-1, 0), (0, 0)],
          },
        ],
        outputs: vec![(0, 0)],
      }),
      _ => Err(format!("Unsupported onnx operation: {op}")),
    }
    .unwrap();

    let mut local_block_idx = vec![];
    let temp = local_graph.basic_blocks;
    local_graph.basic_blocks = vec![];
    for basic_block in temp.into_iter() {
      let name = format!("{basic_block:?}");
      let idx = *basic_blocks_idx.entry(name).or_insert(graph.basic_blocks.len());
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
              if input_tag == "input" {
                (-1, *y)
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
  }

  println!("{graph:?}");
  println!("{:?}", setups.len());

  (graph, setups)
}
