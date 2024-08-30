use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use ark_bn254::Fr;
use ark_std::One;
use ndarray::ArrayD;
use rayon::iter::ParallelIterator;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;

#[derive(Debug)]
pub struct PrecomputedPowBasicBlock {
  pub input_SF: usize,
  pub output_SF: usize,
}
impl BasicBlock for PrecomputedPowBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let base = util::fr_to_int(*inputs[0].first().unwrap());
    let shape = inputs[1].shape();
    let out = util::array_into_iter(inputs[1])
      .map(|x| {
        let mut x = util::fr_to_int(*x) as f32;
        x /= (1 << self.input_SF) as f32;
        x = x.powi(base);
        x *= (1 << self.output_SF) as f32;
        Fr::from(x.round() as i32)
      })
      .collect::<Vec<_>>();
    vec![ArrayD::from_shape_vec(shape, out).unwrap()]
  }
}

pub struct PowLayer;
impl Layer for PowLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    _attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    // if both constants[0] and constants[1] are Some
    if constants[0].is_some() && constants[1].is_some() {
      // if constants[0].len() == 1 and constants[1].len() > 1
      if constants[0].unwrap().0.len() == 1 && constants[1].unwrap().0.len() > 1 {
        let pow = graph.addBB(Box::new(PrecomputedPowBasicBlock {
          input_SF: *onnx::SF_LOG,
          output_SF: *onnx::SF_LOG,
        }));
        let pow_output = graph.addNode(pow, vec![(-1, 0), (-2, 0)]);
        graph.outputs.push((pow_output, 0));
        return (graph, vec![input_shapes[1].clone()], vec![input_types[0]]);
      }
    }
    assert!(constants[1].unwrap().0.len() == 1);
    let N = util::fr_to_int(*constants[1].unwrap().0.first().unwrap());
    assert!(N >= 0);
    if N == 0 {
      let endShape_padded: Vec<usize> = input_shapes[0].clone().iter().map(|&x| util::next_pow(x as u32) as usize).collect();
      let one = graph.addBB(Box::new(ConstOfShapeBasicBlock {
        c: Fr::one(),
        shape: endShape_padded.clone(),
      }));
      let one_output = graph.addNode(one, vec![]);
      graph.outputs.push((one_output, 0));
      return (graph, vec![input_shapes[0].clone()], vec![input_types[0]]);
    }

    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
      N: 1,
    }));
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: *onnx::SF_LOG * 2,
      output_SF: *onnx::SF_LOG,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: *onnx::SF_LOG * 2,
            output_SF: *onnx::SF_LOG,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));
    let mut mul_output = -1;
    let mut change_SF_output = -1;
    for _i in 1..N {
      mul_output = graph.addNode(mul, vec![(-1, 0), (mul_output, 0)]);
      change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    }

    graph.outputs.push((change_SF_output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
