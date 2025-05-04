use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::pad_to_pow_of_two;
use crate::CONFIG;
use ark_bn254::Fr;
use ark_bn254::G1Projective;
use ark_std::{One, Zero};
use ndarray::indices;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::IxDyn;
use ndarray::{concatenate, s, Array1, Array4, ArrayD, Axis, SliceInfo};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;
use tract_onnx::prelude::Outlet; // Import Rayon traits

pub fn slice_nd_array_data(arr: ArrayD<Data>, indices: &[usize]) -> ArrayD<Data> {
  // Create slices from the indices
  let slices: Vec<_> = indices.iter().map(|&i| (0..i).into()).collect();

  // Convert slices into a SliceInfo instance
  let slice_info = unsafe { SliceInfo::<_, IxDyn, IxDyn>::new(slices).unwrap() };

  // Slice the array
  arr.slice_move(slice_info)
}

fn update_kernel(arr: &ArrayD<Fr>, group: usize) -> ArrayD<Fr> {
  let c_in = arr.shape()[0] * group;
  let c_out = arr.shape()[1];
  // arr is [C_in / group, C_out]
  if group == 1 {
    return arr.clone();
  }
  // new_arr is [C_in, C_out]
  let mut new_arr = ArrayD::<Fr>::zeros(IxDyn(&[c_in, c_out]));
  for g in 0..group {
    for i in 0..(c_in / group) {
      for j in 0..(c_out / group) {
        let row = g * (c_in / group) + i;
        let col = g * (c_out / group) + j;
        new_arr[[row, col]] = arr[[i, col]];
      }
    }
  }
  new_arr
}

pub struct Conv2dLayer;

impl Layer for Conv2dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0]; // [1, H_in * W_in, C_in]
    let weight_shape = input_shapes[1]; // [C_out, C_in, k_h, k_w]
    let group = match attributes.iter().find(|x| x.name == "group") {
      Some(v) => v.i as usize,
      None => 1,
    };
    assert!(group * weight_shape[1] == input_shape[2]);

    // only support square image for now
    let H = (input_shape[1] as f64).sqrt() as usize;
    let W = H;

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 2],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 4],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
      Some(v) => v
        .ints
        .iter()
        .map(|x| {
          if *x != 1 {
            panic!("dilations != 1 not supported");
          }
          *x as usize
        })
        .collect(),
      None => vec![1; 2],
    };

    // Conv Add
    let conv_add = graph.addBB(Box::new(Conv2DAddBasicBlock {
      input_shape: vec![H as i32, W as i32],
      kernel_shape: vec![weight_shape[2] as i32, weight_shape[3] as i32],
      stride: strides.clone().into_iter().map(|x| x as i32).collect(),
      padding: pads.clone().into_iter().map(|x| x as i32).collect(),
      out_channels: util::next_pow(weight_shape[0] as u32) as usize,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: util::next_pow(weight_shape[0] as u32) as usize,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let mut cqlin_outputs = Vec::new();
    for k_h in 0..weight_shape[2] {
      for k_w in 0..weight_shape[3] {
        let k = constants[1].unwrap().0.slice(s![.., .., k_h, k_w]);
        let k = update_kernel(&k.t().to_owned().into_dyn(), group);
        let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
          basic_block: Box::new(CQLinBasicBlock {
            setup: k, // [C_in, C_out]
          }),
          N: 2,
        }));
        let cqlin_output = graph.addNode(cqlin, vec![(-1, 0)]);
        cqlin_outputs.push((cqlin_output, 0)); // [1, H_in * W_in, C_out]
      }
    }
    let conv_output = graph.addNode(conv_add, cqlin_outputs); // [1, H_o * W_o, C_out]

    // Change SF
    let mut output = graph.addNode(change_SF, vec![(conv_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(conv_output, 0), (output, 0)]);

    if input_shapes.len() > 2 {
      output = graph.addNode(add, vec![(output, 0), (-3, 0)]);
    }

    let H_o = (H - weight_shape[2] + pads[0] + pads[2]) / strides[0] + 1;
    let W_o = (W - weight_shape[3] + pads[1] + pads[3]) / strides[1] + 1;

    let output_shape = vec![1, H_o * W_o, weight_shape[0]];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

#[derive(Debug)]
pub struct MultiHeadConv2dAggBasicBlock {
  pub output_shape: Vec<usize>, // [1, H_out * W_out, head_dim]
}
impl BasicBlock for MultiHeadConv2dAggBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 9);
    let mut final_output_shape = self.output_shape.clone();
    final_output_shape[1] *= 9;

    let mut result = ArrayD::<Fr>::zeros(IxDyn(&final_output_shape));
    for head in 0..9 {
      let input_slice = util::slice_nd_array(inputs[head].clone(), &self.output_shape);
      result.slice_axis_mut(Axis(1), (head * self.output_shape[1]..(head + 1) * self.output_shape[1]).into()).assign(&input_slice);
    }
    result = util::pad_to_pow_of_two(&result, &Fr::zero());

    Ok(vec![result])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let mut final_output_shape = vec![self.output_shape[0], self.output_shape[1]];
    final_output_shape[1] *= 9;
    let data_zero = Data {
      raw: vec![Fr::zero(); self.output_shape[2]],
      poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
      r: Fr::zero(),
      g1: G1Projective::zero(),
    };
    let mut result = ArrayD::from_shape_fn(IxDyn(&final_output_shape), |_| data_zero.clone());
    for head in 0..9 {
      let input_slice = slice_nd_array_data(inputs[head].clone(), &[self.output_shape[0], self.output_shape[1]]);
      result.slice_axis_mut(Axis(1), (head * self.output_shape[1]..(head + 1) * self.output_shape[1]).into()).assign(&input_slice);
    }
    result = util::pad_to_pow_of_two(&result, &data_zero);
    vec![result]
  }
}

pub struct MultiHeadConv2dLayer;

impl Layer for MultiHeadConv2dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0]; // [1, H_in * W_in, C_in]
    let weight_shape = input_shapes[1]; // [C_out, C_in, k_h, k_w]

    // only support square image for now
    let H = (input_shape[1] as f64).sqrt() as usize;
    let W = H;

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 2],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 4],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
      Some(v) => v
        .ints
        .iter()
        .map(|x| {
          if *x != 1 {
            panic!("dilations != 1 not supported");
          }
          *x as usize
        })
        .collect(),
      None => vec![1; 2],
    };
    let head_dim = match attributes.iter().find(|x| x.name == "head_dim") {
      Some(v) => v.i,
      None => panic!("head_dim not found"),
    };
    let head_dim = head_dim as usize;
    let num_heads = 9; // RetinaNet uses 9 heads

    // Conv Add
    let conv_add = graph.addBB(Box::new(Conv2DAddBasicBlock {
      input_shape: vec![H as i32, W as i32],
      kernel_shape: vec![weight_shape[2] as i32, weight_shape[3] as i32],
      stride: strides.clone().into_iter().map(|x| x as i32).collect(),
      padding: pads.clone().into_iter().map(|x| x as i32).collect(),
      out_channels: util::next_pow(head_dim as u32) as usize,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: util::next_pow(head_dim as u32) as usize,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let mut head_outputs = Vec::new();
    for head in 0..num_heads {
      let mut cqlin_outputs = Vec::new();
      for k_h in 0..weight_shape[2] {
        for k_w in 0..weight_shape[3] {
          let k = constants[1].unwrap().0.slice(s![(head * head_dim)..((head + 1) * head_dim), .., k_h, k_w]);
          let k = k.t().to_owned().into_dyn();
          let k = util::pad_to_pow_of_two(&k, &Fr::zero());
          let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
            basic_block: Box::new(CQLinBasicBlock {
              setup: k, // [C_in, C_out]
            }),
            N: 2,
          }));
          let cqlin_output = graph.addNode(cqlin, vec![(-1, 0)]);
          cqlin_outputs.push((cqlin_output, 0)); // [1, H_in * W_in, C_out]
        }
      }
      let conv_output = graph.addNode(conv_add, cqlin_outputs); // [1, H_o * W_o, C_out]

      // Change SF
      let mut output = graph.addNode(change_SF, vec![(conv_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(conv_output, 0), (output, 0)]);

      if input_shapes.len() > 2 {
        let b = constants[2].unwrap().0.slice(s![(head * head_dim)..((head + 1) * head_dim)]);
        let b = b.to_owned().into_dyn();
        let b = util::pad_to_pow_of_two(&b, &Fr::zero());
        let bias = graph.addBB(Box::new(Const2BasicBlock { c: b }));
        let bias_output = graph.addNode(bias, vec![]);
        output = graph.addNode(add, vec![(output, 0), (bias_output, 0)]);
      }
      head_outputs.push((output, 0));
    }

    let H_o = (H - weight_shape[2] + pads[0] + pads[2]) / strides[0] + 1;
    let W_o = (W - weight_shape[3] + pads[1] + pads[3]) / strides[1] + 1;

    let output = graph.addBB(Box::new(MultiHeadConv2dAggBasicBlock {
      output_shape: vec![1, H_o * W_o, head_dim],
    }));
    let output = graph.addNode(output, head_outputs);

    let output_shape = vec![1, num_heads * H_o * W_o, head_dim];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

pub struct Conv3dLayer;

impl Layer for Conv3dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0]; // [1, D1_in * D2_in * D3_in, C_in]
    let weight_shape = input_shapes[1]; // [C_out, C_in, D1, D2, D3]

    // only support cube image for now
    let D = (input_shape[1] as f64).cbrt() as usize;

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 3],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 6],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
      Some(v) => v
        .ints
        .iter()
        .map(|x| {
          if *x != 1 {
            panic!("dilations != 1 not supported");
          }
          *x as usize
        })
        .collect(),
      None => vec![1; 3],
    };

    // Conv Add
    let conv_add = graph.addBB(Box::new(Conv3DAddBasicBlock {
      input_shape: vec![D as i32, D as i32, D as i32],
      kernel_shape: vec![weight_shape[2] as i32, weight_shape[3] as i32, weight_shape[4] as i32],
      stride: strides.clone().into_iter().map(|x| x as i32).collect(),
      padding: pads.clone().into_iter().map(|x| x as i32).collect(),
      out_channels: util::next_pow(weight_shape[0] as u32) as usize,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: util::next_pow(weight_shape[0] as u32) as usize,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let mut cqlin_outputs = Vec::new();
    for k_d1 in 0..weight_shape[2] {
      for k_d2 in 0..weight_shape[3] {
        for k_d3 in 0..weight_shape[4] {
          let k = constants[1].unwrap().0.slice(s![.., .., k_d1, k_d2, k_d3]);
          let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
            basic_block: Box::new(CQLinBasicBlock {
              setup: k.t().to_owned().into_dyn(), // [C_in, C_out]
            }),
            N: 2,
          }));
          let cqlin_output = graph.addNode(cqlin, vec![(-1, 0)]);
          cqlin_outputs.push((cqlin_output, 0)); // [1, D_in * D_in * D_in, C_out]
        }
      }
    }
    let conv_output = graph.addNode(conv_add, cqlin_outputs); // [1, D_o * D_o * D_o, C_out]

    // Change SF
    let mut output = graph.addNode(change_SF, vec![(conv_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(conv_output, 0), (output, 0)]);
    if input_shapes.len() > 2 {
      output = graph.addNode(add, vec![(output, 0), (-3, 0)]);
    }

    let D1_o = (D - weight_shape[2] + pads[0] + pads[3]) / strides[0] + 1;
    let D2_o = (D - weight_shape[3] + pads[1] + pads[4]) / strides[1] + 1;
    let D3_o = (D - weight_shape[4] + pads[2] + pads[5]) / strides[2] + 1;

    let output_shape = vec![1, D1_o * D2_o * D3_o, weight_shape[0]];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

pub struct ConcatConv3dLayer;

impl Layer for ConcatConv3dLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0]; // [1, D1_in * D2_in * D3_in, C_in]
    let input_shape_1 = input_shapes[1];
    let in_channels = input_shape[2];
    assert!(input_shape[1] == input_shape_1[1]);
    let weight_shape = input_shapes[2]; // [C_out, C_in, D1, D2, D3]

    // only support cube image for now
    let D = (input_shape[1] as f64).cbrt() as usize;

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 3],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 6],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
      Some(v) => v
        .ints
        .iter()
        .map(|x| {
          if *x != 1 {
            panic!("dilations != 1 not supported");
          }
          *x as usize
        })
        .collect(),
      None => vec![1; 3],
    };

    // Conv Add
    let conv_add = graph.addBB(Box::new(Conv3DAddBasicBlock {
      input_shape: vec![D as i32, D as i32, D as i32],
      kernel_shape: vec![weight_shape[2] as i32, weight_shape[3] as i32, weight_shape[4] as i32],
      stride: strides.clone().into_iter().map(|x| x as i32).collect(),
      padding: pads.clone().into_iter().map(|x| x as i32).collect(),
      out_channels: util::next_pow(weight_shape[0] as u32) as usize,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: util::next_pow(weight_shape[0] as u32) as usize,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let mut cqlin_outputs = Vec::new();
    for k_d1 in 0..weight_shape[2] {
      for k_d2 in 0..weight_shape[3] {
        for k_d3 in 0..weight_shape[4] {
          let mut add_outputs = Vec::new();
          for i in 0..2 {
            let k = constants[2].unwrap().0.slice(s![.., (in_channels * i)..(in_channels * (i + 1)), k_d1, k_d2, k_d3]);
            let k = util::pad_to_pow_of_two(&k.into_dyn().to_owned(), &Fr::zero());
            let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
              basic_block: Box::new(CQLinBasicBlock {
                setup: k.t().to_owned(), // [C_in, C_out]
              }),
              N: 2,
            }));
            let cqlin_output = graph.addNode(cqlin, vec![(-1 - i as i32, 0)]);
            add_outputs.push((cqlin_output, 0)); // 2*[1, D_in * D_in * D_in, C_out]
          }
          let cqlin_output = graph.addNode(add, add_outputs); //  [1, D_in * D_in * D_in, C_out]

          cqlin_outputs.push((cqlin_output, 0)); // [1, D_in * D_in * D_in, C_out]
        }
      }
    }
    let conv_output = graph.addNode(conv_add, cqlin_outputs); // [1, D_o * D_o * D_o, C_out]

    // Change SF
    let output = graph.addNode(change_SF, vec![(conv_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(conv_output, 0), (output, 0)]);

    let D1_o = (D - weight_shape[2] + pads[0] + pads[3]) / strides[0] + 1;
    let D2_o = (D - weight_shape[3] + pads[1] + pads[4]) / strides[1] + 1;
    let D3_o = (D - weight_shape[4] + pads[2] + pads[5]) / strides[2] + 1;

    let output_shape = vec![1, D1_o * D2_o * D3_o, weight_shape[0]];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}

pub struct Conv3dTransposeLayer;
impl Layer for Conv3dTransposeLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();
    let input_shape = input_shapes[0]; // [1, D1_in * D2_in * D3_in, C_in]
    let weight_shape = input_shapes[1]; // [C_in, C_out, D1, D2, D3]

    // only support cube image for now
    let D = (input_shape[1] as f64).cbrt() as usize;

    let strides = match attributes.iter().find(|x| x.name == "strides") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![1; 3],
    };
    let pads = match attributes.iter().find(|x| x.name == "pads") {
      Some(v) => v.ints.iter().map(|x| *x as usize).collect(),
      None => vec![0; 6],
    };
    let _dilations = match attributes.iter().find(|x| x.name == "dilations") {
      Some(v) => v
        .ints
        .iter()
        .map(|x| {
          if *x != 1 {
            panic!("dilations != 1 not supported");
          }
          *x as usize
        })
        .collect(),
      None => vec![1; 3],
    };

    // Conv Add
    let conv_trans = graph.addBB(Box::new(Conv3DTransposeBasicBlock {
      input_shape: vec![D as i32, D as i32, D as i32],
      kernel_shape: vec![weight_shape[2] as i32, weight_shape[3] as i32, weight_shape[4] as i32],
      stride: strides.clone().into_iter().map(|x| x as i32).collect(),
      padding: pads.clone().into_iter().map(|x| x as i32).collect(),
      out_channels: util::next_pow(weight_shape[1] as u32) as usize,
    }));

    // Add bias
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));

    // Scale down
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: util::next_pow(weight_shape[1] as u32) as usize,
        setup: Some((
          Box::new(ChangeSFBasicBlock {
            input_SF: sf_log * 2,
            output_SF: sf_log,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let mut cqlin_outputs = Vec::new();
    for k_d1 in 0..weight_shape[2] {
      for k_d2 in 0..weight_shape[3] {
        for k_d3 in 0..weight_shape[4] {
          let k = constants[1].unwrap().0.slice(s![.., .., k_d1, k_d2, k_d3]);
          let cqlin = graph.addBB(Box::new(RepeaterBasicBlock {
            basic_block: Box::new(CQLinBasicBlock {
              setup: k.to_owned().into_dyn(), // [C_in, C_out]
            }),
            N: 2,
          }));
          let cqlin_output = graph.addNode(cqlin, vec![(-1, 0)]);
          cqlin_outputs.push((cqlin_output, 0)); // [1, D_in * D_in * D_in, C_out]
        }
      }
    }
    let conv_output = graph.addNode(conv_trans, cqlin_outputs); // [1, D_o * D_o * D_o, C_out]

    // Change SF
    let mut output = graph.addNode(change_SF, vec![(conv_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(conv_output, 0), (output, 0)]);
    if input_shapes.len() > 2 {
      output = graph.addNode(add, vec![(output, 0), (-3, 0)]);
    }
    let D1_o = (D - 1) * strides[0] - pads[0] - pads[3] + (weight_shape[2] - 1) + 1;
    let D2_o = (D - 1) * strides[1] - pads[1] - pads[4] + (weight_shape[3] - 1) + 1;
    let D3_o = (D - 1) * strides[2] - pads[2] - pads[5] + (weight_shape[4] - 1) + 1;

    let output_shape = vec![1, D1_o * D2_o * D3_o, weight_shape[1]];
    graph.outputs.push((output, 0));
    (graph, vec![output_shape], vec![input_types[0]])
  }
}
