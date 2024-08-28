use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct ClipLayer;
impl Layer for ClipLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let min = util::fr_to_int(constants[1].unwrap().0.as_slice().unwrap()[0]) as f32;
    let max = util::fr_to_int(constants[2].unwrap().0.as_slice().unwrap()[0]) as f32;

    let clip = graph.addBB(Box::new(ClipBasicBlock { min: min, max: max }));
    let clip_output = graph.addNode(clip, vec![(-1, 0)]);
    let clip_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((Box::new(ClipBasicBlock { min: min, max: max }), *onnx::CQ_RANGE_LOWER, *onnx::CQ_RANGE)),
      }),
      N: 1,
    }));
    let _ = graph.addNode(clip_check, vec![(-1, 0), (clip_output, 0)]);

    graph.outputs.push((clip_output, 0));

    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
