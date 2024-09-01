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
    let base = util::fr_to_int(*inputs[0].first().unwrap()) as f32;
    let shape = inputs[1].shape();
    let out = util::array_into_iter(inputs[1])
      .map(|x| {
        let mut b = base;
        b /= (1 << self.input_SF) as f32;
        b = b.powi(util::fr_to_int(*x));
        b *= (1 << self.output_SF) as f32;
        Fr::from(b.round() as i32)
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
    assert!(constants[1].is_some());

    // Note: the following code is a workaround for the case that constants[0] is a scalar and constants[1] is a tensor
    //       If we want to formally prove this, we need to
    //       (1) either implement a new basic block that can handle this case
    //       (2) or perform element-wise pow and copy the result to the output tensor
    //       both of which are a little bit complicated.
    //       Fortunately, this case only happens in the precomputable part of RoPE embedding for now.
    //       So, we can just use a simple basic block that can handle this case without proving.
    // TODO: think about how to handle this case in a more general way later
    // if both constants[0] and constants[1] are Some
    if constants[0].is_some() && constants[1].is_some() {
      // if constants[0].len() == 1 and constants[1].len() > 1
      if constants[0].unwrap().0.len() == 1 && constants[1].unwrap().0.len() > 1 {
        let c_vec = match constants[1].unwrap().1 {
          DatumType::I32 | DatumType::I64 => constants[1].unwrap().0.iter().map(|x| *x).collect::<Vec<_>>(),
          DatumType::F32 => {
            constants[1].unwrap().0.iter().map(|x| Fr::from((util::fr_to_int(*x) as f32 / *onnx::SF_FLOAT) as i32)).collect::<Vec<_>>()
          }
          _ => panic!("unsupported type"),
        };
        let shape = constants[1].unwrap().0.shape();
        let const2 = graph.addBB(Box::new(Const2BasicBlock {
          c: ArrayD::from_shape_vec(shape, c_vec).unwrap(),
        }));
        let pow = graph.addBB(Box::new(PrecomputedPowBasicBlock {
          input_SF: *onnx::SF_LOG,
          output_SF: *onnx::SF_LOG,
        }));
        let const2_output = graph.addNode(const2, vec![]);
        let pow_output = graph.addNode(pow, vec![(-1, 0), (const2_output, 0)]);
        graph.outputs.push((pow_output, 0));
        return (graph, vec![input_shapes[1].clone()], vec![input_types[0]]);
      }
    }

    assert!(constants[1].unwrap().0.len() == 1);
    let N = match constants[1].unwrap().1 {
      DatumType::I32 | DatumType::I64 => util::fr_to_int(*constants[1].unwrap().0.first().unwrap()),
      DatumType::F32 => (util::fr_to_int(*constants[1].unwrap().0.first().unwrap()) as f32 / *onnx::SF_FLOAT) as i32,
      _ => panic!("unsupported type"),
    };

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
    // TODO: when N > 2, it is better to use a more efficient way to calculate the power such as the way in nonlinear.rs
    for _i in 1..N {
      mul_output = graph.addNode(mul, vec![(-1, 0), (mul_output, 0)]);
      change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    }

    graph.outputs.push((change_SF_output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
