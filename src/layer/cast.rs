use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::datumtype_to_sf;
use ark_bn254::Fr;
use ndarray::ArrayD;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

pub struct CastLayer;
impl Layer for CastLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let to = match attributes.iter().filter(|x| x.name == "to").next() {
      Some(v) => vec![util::datatype_to_datumtype(v.i as i32)],
      None => vec![input_types[0]],
    };
    let input_SF = datumtype_to_sf(input_types[0]);
    let output_SF = datumtype_to_sf(to[0]);
    let id = if input_SF == output_SF {
      graph.addBB(Box::new(IdBasicBlock {}))
    } else {
      graph.addBB(Box::new(ChangeSFBasicBlock { input_SF, output_SF }))
    };
    let change_sf_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: input_SF,
            output_SF: output_SF,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let id_output = graph.addNode(id, vec![(-1, 0)]);
    if input_SF != output_SF {
      let _ = graph.addNode(change_sf_check, vec![(-1, 0), (id_output, 0)]);
    }
    graph.outputs.push((id_output, 0));
    (graph, vec![input_shapes[0].clone()], to)
  }
}
