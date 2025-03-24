use crate::basic_block::*;
use crate::graph::*;
use crate::layer::squeeze::UnsqueezeBasicBlock;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
use crate::util::CQArrayType;
use ark_bn254::Fr;
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::DatumType;
use util::copy_constraint::get_reshape_indices;

// BatchNormLayer is a struct that represents a batch normalization layer, which computes
// Y = (X - input_mean) * scale / sqrt(input_var + epsilon) + bias
pub struct BatchNormLayer;
impl Layer for BatchNormLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let X_shape = input_shapes[0];
    let scale_shape = input_shapes[1];
    let bias_shape = input_shapes[2];
    let mean_shape = input_shapes[3];
    let var_shape = input_shapes[4];

    // Check that the shapes are correct
    // X: [N, C, D1, D2, ..., DN]
    // scale: [C]
    // bias: [C]
    // mean: [C]
    // var: [C]
    assert!(X_shape[1] == scale_shape[0] && scale_shape[0] == bias_shape[0] && bias_shape[0] == mean_shape[0] && mean_shape[0] == var_shape[0]);
    assert!(scale_shape.len() == 1 && bias_shape.len() == 1 && mean_shape.len() == 1 && var_shape.len() == 1);

    let training_mode_attr = attributes.iter().filter(|x| x.name == "training_mode").next();
    let training_mode = if let Some(x) = training_mode_attr {
      // training_mode is provided
      x.i as i8
    } else {
      // training_mode is not provided
      0
    };
    // we only support training_mode = 0 (inference) for now
    assert!(training_mode == 0);

    let epsilon_attr = attributes.iter().filter(|x| x.name == "epsilon").next();
    let mut epsilon = if let Some(x) = epsilon_attr {
      // epsilon is provided
      x.f as f32
    } else {
      // epsilon is not provided, use the default value
      1e-5
    };
    // epsilon *= onnx::SF_FLOAT.read().unwrap().to_owned();
    epsilon = 1.0;

    let epsilon = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(epsilon.round() as i32)]).into_dyn(),
    }));
    let scale_shape_padded = util::next_pow(scale_shape[0] as u32) as usize;
    let reshape_1 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![1, scale_shape_padded],
    }));
    let permutation = ((0..scale_shape_padded).collect(), vec![0]);
    let permute = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock {
        permutation: permutation,
        n: 1,
        m: scale_shape_padded,
      }),
      N: 2,
    }));
    let num_one = X_shape.len() - 2;
    let reshape_2 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![scale_shape_padded].into_iter().chain(vec![1; num_one]).collect(),
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: scale_shape_padded.next_power_of_two(),
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

    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock {
        output_SF: onnx::SF.read().unwrap().to_owned(),
      }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: scale_shape_padded,
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock {
        c: onnx::SF.read().unwrap().to_owned() * 2,
      }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));

    let split_ind = vec![1; util::next_pow(scale_shape[0] as u32) as usize];
    let split = graph.addBB(Box::new(SplitBasicBlock {
      axis: 0,
      split: split_ind.clone(),
    }));
    let split_x = graph.addBB(Box::new(SplitBasicBlock { axis: 1, split: split_ind }));
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 1 }));
    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: sf_log,
      output_SF: sf_log,
    }));
    let sf = onnx::SF.read().unwrap().to_owned();
    let sf_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(sf as i32)]).into_dyn(),
    }));
    let two_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(2)]).into_dyn(),
    }));
    let len = util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize;
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock { len }),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let non_negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: scale_shape_padded,
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: scale_shape_padded,
        setup: util::CQArrayType::Negative,
      }),
      N: 1,
    }));

    // output epsilon
    let epsilon_output = graph.addNode(epsilon, vec![]);

    // reshape scale
    let scale_temp_output = graph.addNode(reshape_1, vec![(-2, 0)]);
    let scale_temp_output = graph.addNode(permute, vec![(scale_temp_output, 0)]);
    let scale_output = graph.addNode(reshape_2, vec![(scale_temp_output, 0)]);

    // reshape bias
    let bias_temp_output = graph.addNode(reshape_1, vec![(-3, 0)]);
    let bias_temp_output = graph.addNode(permute, vec![(bias_temp_output, 0)]);
    let bias_output = graph.addNode(reshape_2, vec![(bias_temp_output, 0)]);

    // reshape mean
    let mean_temp_output = graph.addNode(reshape_1, vec![(-4, 0)]);
    let mean_temp_output = graph.addNode(permute, vec![(mean_temp_output, 0)]);
    let mean_output = graph.addNode(reshape_2, vec![(mean_temp_output, 0)]);

    // reshape var
    let var_temp_output = graph.addNode(reshape_1, vec![(-5, 0)]);
    let var_temp_output = graph.addNode(permute, vec![(var_temp_output, 0)]);
    let var_output = graph.addNode(reshape_2, vec![(var_temp_output, 0)]);

    // Step 1. X - mean
    let sub_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]);

    // Step 2. var + epsilon
    let add_output = graph.addNode(add, vec![(var_output, 0), (epsilon_output, 0)]);

    // Step 3. sqrt(var + epsilon)
    // x = var + epsilon
    // SqrtBB(x) = sqrt(x/SF)*SF + eps (where -1 < eps < 1)
    let sqrt_output = graph.addNode(sqrt, vec![(add_output, 0)]);
    // The following operations are to check if sqrt_output is correct
    // square_sqrt = SqrtBB(x)^2 = x*SF + 2*sqrt(x/SF)*SF*eps + eps^2
    let square_sqrt = graph.addNode(mul, vec![(sqrt_output, 0), (sqrt_output, 0)]);
    // scale_input_by_sf = x*SF
    let sf_const_output = graph.addNode(sf_const, vec![]);
    let scale_input_by_sf = graph.addNode(mul_scalar, vec![(add_output, 0), (sf_const_output, 0)]);
    // difference = SqrtBB(x)^2 - x*SF = 2*sqrt(x/SF)*SF*eps + eps^2 = 2*SqrtBB(x)*eps + eps^2
    // Because -1 < eps < 1, -2*SqrtBB(x) < 2*SqrtBB(x)*eps < 2*SqrtBB(x) and 0 < eps^2 < 1.

    // Therefore, - 2*SqrtBB(x) < difference < 2*SqrtBB(x) + 1.
    // The following two inequalities should hold:
    // 1. difference + 2*SqrtBB(x) >= 0
    // 2. difference - 2*SqrtBB(x) - 1 < 0
    let difference = graph.addNode(sub, vec![(square_sqrt, 0), (scale_input_by_sf, 0)]);
    // scale_output_by_2 = 2*SqrtBB(x)
    let two_const_output = graph.addNode(two_const, vec![]);
    let scale_output_by_2 = graph.addNode(mul_scalar, vec![(sqrt_output, 0), (two_const_output, 0)]);
    let d_plus_scale_output_by_2 = graph.addNode(add, vec![(difference, 0), (scale_output_by_2, 0)]);
    // d_minus_scale_output_by_2 = difference - 2*SqrtBB(x)
    let d_minus_scale_output_by_2 = graph.addNode(sub, vec![(difference, 0), (scale_output_by_2, 0)]);
    let _ = graph.addNode(non_negative_check, vec![(d_plus_scale_output_by_2, 0)]);
    let _ = graph.addNode(negative_check, vec![(d_minus_scale_output_by_2, 0)]);

    // Step 4. (X - mean) / sqrt(var + epsilon)
    let split_sub_output = graph.addNode(split_x, vec![(sub_output, 0)]);
    let split_sqrt_output = graph.addNode(split, vec![(sqrt_output, 0)]);
    let split_scale_output = graph.addNode(split, vec![(scale_output, 0)]);
    let mut tmp_outputs = vec![];
    // here we perform the batch normalization for each channel
    for i in 0..util::next_pow(scale_shape[0] as u32) as usize {
      let div_output = graph.addNode(div, vec![(split_sub_output, i), (split_sqrt_output, i)]);
      let a_SF2 = graph.addNode(mul_SF2, vec![(split_sub_output, i)]);
      let a_SF2_plus_b = graph.addNode(add, vec![(a_SF2, 0), (split_sqrt_output, i)]);
      let b2 = graph.addNode(mul_2, vec![(split_sqrt_output, i)]);
      let qb2 = graph.addNode(mul_scalar, vec![(div_output, 0), (b2, 0)]);
      let qb2_plus_r = graph.addNode(add, vec![(qb2, 0), (div_output, 1)]);
      let _ = graph.addNode(eq, vec![(a_SF2_plus_b, 0), (qb2_plus_r, 0)]);
      // Now check r≥0:
      let _ = graph.addNode(range_check, vec![(div_output, 1)]);
      // Now check 2b-r≥0:
      let b2_minus_r = graph.addNode(sub, vec![(b2, 0), (div_output, 1)]);
      let _ = graph.addNode(range_check, vec![(b2_minus_r, 0)]);

      // Step 5. scale * (X - mean) / sqrt(var + epsilon)
      let mul_output = graph.addNode(mul_scalar, vec![(div_output, 0), (split_scale_output, i)]);
      let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
      let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);

      tmp_outputs.push((change_SF_output, 0));
    }
    let concat_output = graph.addNode(concat, tmp_outputs);

    // Step 6. scale * (X - mean) / sqrt(var + epsilon) + bias
    let output = graph.addNode(add, vec![(concat_output, 0), (bias_output, 0)]);

    graph.outputs.push((output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}

// InstanceNorm is a struct that represents an instance normalization layer, which computes
// Y = (X - mean) * scale / sqrt(var + epsilon) + bias,
// where mean and var are computed per instance per channel
pub struct InstanceNormLayer;
impl Layer for InstanceNormLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let X_shape = input_shapes[0];
    let scale_shape = input_shapes[1];
    let bias_shape = input_shapes[2];

    // Check that the shapes are correct
    // X: [N, C, D1, D2, ..., DN]
    // scale: [C]
    // bias: [C]
    assert!(X_shape[1] == scale_shape[0] && scale_shape[0] == bias_shape[0]);
    assert!(scale_shape.len() == 1 && bias_shape.len() == 1);

    let epsilon_attr = attributes.iter().filter(|x| x.name == "epsilon").next();
    let mut epsilon = if let Some(x) = epsilon_attr {
      // epsilon is provided
      x.f as f32
    } else {
      // epsilon is not provided, use the default value
      1e-5
    };
    epsilon *= onnx::SF_FLOAT.read().unwrap().to_owned();

    // X_shape_for_mean: [N, C, D1 * D2 * ... * DN]
    let len = X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y);
    let x_shape_for_mean = vec![X_shape[0], X_shape[1], len];
    let permutation = get_reshape_indices(X_shape.to_vec(), x_shape_for_mean);
    let padded_input_shape: Vec<_> = X_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&padded_input_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let len = util::next_pow(len as u32) as usize;
    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock { len }),
      N: 1,
    }));
    let div_const = graph.addBB(Box::new(DivConstProofBasicBlock {
      c: X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y as u32),
    }));
    let div_const_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: 1,
        setup: CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    let mut mean_shape_padded = vec![util::next_pow(X_shape[0] as u32) as usize, util::next_pow(X_shape[1] as u32) as usize];
    mean_shape_padded = mean_shape_padded.into_iter().chain(vec![1; X_shape.len() - 2]).collect();
    let reshape_0 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: mean_shape_padded.clone(),
    }));

    let epsilon = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(epsilon.round() as i32)]).into_dyn(),
    }));
    let scale_shape_padded = util::next_pow(scale_shape[0] as u32) as usize;
    let reshape_1 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![1, scale_shape_padded],
    }));
    let permutation = ((0..scale_shape_padded).collect(), vec![0]);
    let permute = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock {
        permutation: permutation,
        n: 1,
        m: scale_shape_padded,
      }),
      N: 2,
    }));
    let num_one = X_shape.len() - 2;
    let reshape_2 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![scale_shape_padded].into_iter().chain(vec![1; num_one]).collect(),
    }));
    let len = util::next_pow(input_shapes[0][input_shapes[0].len() - 1] as u32) as usize;
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock { len }),
      N: 1,
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: X_shape[X_shape.len() - 1].next_power_of_two(),
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
    let div_SF = graph.addBB(Box::new(DivConstBasicBlock {
      c: (X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y) as f32).sqrt() * ((1 << sf_log) as f32),
    }));
    let div_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: X_shape[X_shape.len() - 1].next_power_of_two(),
        setup: Some((
          Box::new(DivConstBasicBlock {
            c: (X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y) as f32).sqrt() * ((1 << sf_log) as f32),
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let sf = onnx::SF.read().unwrap().to_owned();
    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock { output_SF: sf }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: X_shape[X_shape.len() - 1].next_power_of_two(),
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: sf * 2 }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));

    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: sf_log,
      output_SF: sf_log,
    }));
    let sf = onnx::SF.read().unwrap().to_owned();
    let sf_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(sf as i32)]).into_dyn(),
    }));
    let two_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(2)]).into_dyn(),
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let non_negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: 1,
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: 1,
        setup: util::CQArrayType::Negative,
      }),
      N: 1,
    }));

    // Reshape X to [N * C, D1 * D2 * ... * DN]
    let shape_for_split_x: Vec<_> = vec![mean_shape_padded[0] * mean_shape_padded[1]]
      .into_iter()
      .chain(X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().map(|x| util::next_pow(*x as u32) as usize))
      .collect();
    let reshape_3 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: shape_for_split_x.clone(),
    }));
    // Reshape var + epsilon to [N * C, 1, ..., 1]
    let shape_for_split_var = vec![mean_shape_padded[0] * mean_shape_padded[1]].into_iter().chain(vec![1; X_shape.len() - 2]).collect();
    let reshape_4 = graph.addBB(Box::new(ReshapeBasicBlock { shape: shape_for_split_var }));
    let split_ind = vec![1; util::next_pow(scale_shape[0] as u32) as usize];
    let split = graph.addBB(Box::new(SplitBasicBlock { axis: 0, split: split_ind }));
    let split_x_ind = vec![1; util::next_pow(shape_for_split_x[0] as u32) as usize];
    let split_x = graph.addBB(Box::new(SplitBasicBlock { axis: 0, split: split_x_ind }));
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 0 }));
    let reshape_concat = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: X_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect(),
    }));

    // Step 0. Compute epsilon, mean, var, scale, and bias
    // output epsilon
    let epsilon_output = graph.addNode(epsilon, vec![]);

    // compute mean (shape: [N, C, 1, ..., 1])
    let cc_output = graph.addNode(cc, vec![(-1, 0)]);
    let sum_output = graph.addNode(sum, vec![(cc_output, 0)]);
    let mean_output = graph.addNode(div_const, vec![(sum_output, 0)]);
    let _ = graph.addNode(div_const_check, vec![(mean_output, 1)]);
    let _ = graph.addNode(div_const_check, vec![(mean_output, 2)]);
    let mean_output = graph.addNode(reshape_0, vec![(mean_output, 0)]);

    // compute var (shape: [N, C, 1, ..., 1])
    // m = mean(X)
    // s = size(X)
    // X - m
    let sub_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]);
    // (X - m) * SF / (sqrt(s) * SF)
    let div_SF_output = graph.addNode(div_SF, vec![(sub_output, 0)]);
    let _ = graph.addNode(div_SF_check, vec![(sub_output, 0), (div_SF_output, 0)]);
    // (X - m)^2 / s
    let mul_output = graph.addNode(mul, vec![(sub_output, 0), (sub_output, 0)]);
    let cc_output = graph.addNode(cc, vec![(mul_output, 0)]);
    // SUM((X - m)^2) / s
    let var_output = graph.addNode(sum, vec![(cc_output, 0)]);
    let var_output = graph.addNode(reshape_0, vec![(var_output, 0)]);

    // reshape scale (shape: [C, 1, ..., 1])
    let scale_temp_output = graph.addNode(reshape_1, vec![(-2, 0)]);
    let scale_temp_output = graph.addNode(permute, vec![(scale_temp_output, 0)]);
    let scale_output = graph.addNode(reshape_2, vec![(scale_temp_output, 0)]);

    // reshape bias (shape: [C, 1, ..., 1])
    let bias_temp_output = graph.addNode(reshape_1, vec![(-3, 0)]);
    let bias_temp_output = graph.addNode(permute, vec![(bias_temp_output, 0)]);
    let bias_output = graph.addNode(reshape_2, vec![(bias_temp_output, 0)]);

    // Step 1. X - mean (shape: [N * C, D1, D2, ..., DN])
    let sub_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]);
    let x_minus_mean_output = graph.addNode(reshape_3, vec![(sub_output, 0)]);

    // Step 2. var + epsilon (shape: [N * C, 1, ..., 1])
    let add_output = graph.addNode(add, vec![(var_output, 0), (epsilon_output, 0)]);
    let var_plus_eps_output = graph.addNode(reshape_4, vec![(add_output, 0)]);

    // Step 3. sqrt(var + epsilon) (shape: [N * C, 1, ..., 1])
    // x = var + epsilon
    // SqrtBB(x) = sqrt(x/SF)*SF + eps (where -1 < eps < 1)
    let sqrt_output = graph.addNode(sqrt, vec![(var_plus_eps_output, 0)]);
    // The following operations are to check if sqrt_output is correct
    // square_sqrt = SqrtBB(x)^2 = x*SF + 2*sqrt(x/SF)*SF*eps + eps^2
    let square_sqrt = graph.addNode(mul, vec![(sqrt_output, 0), (sqrt_output, 0)]);
    // scale_input_by_sf = x*SF
    let sf_const_output = graph.addNode(sf_const, vec![]);
    let scale_input_by_sf = graph.addNode(mul_scalar, vec![(var_plus_eps_output, 0), (sf_const_output, 0)]);
    // difference = SqrtBB(x)^2 - x*SF = 2*sqrt(x/SF)*SF*eps + eps^2 = 2*SqrtBB(x)*eps + eps^2
    // Because -1 < eps < 1, -2*SqrtBB(x) < 2*SqrtBB(x)*eps < 2*SqrtBB(x) and 0 < eps^2 < 1.

    // Therefore, - 2*SqrtBB(x) < difference < 2*SqrtBB(x) + 1.
    // The following two inequalities should hold:
    // 1. difference + 2*SqrtBB(x) >= 0
    // 2. difference - 2*SqrtBB(x) - 1 < 0
    let difference = graph.addNode(sub, vec![(square_sqrt, 0), (scale_input_by_sf, 0)]);
    // scale_output_by_2 = 2*SqrtBB(x)
    let two_const_output = graph.addNode(two_const, vec![]);
    let scale_output_by_2 = graph.addNode(mul_scalar, vec![(sqrt_output, 0), (two_const_output, 0)]);
    let d_plus_scale_output_by_2 = graph.addNode(add, vec![(difference, 0), (scale_output_by_2, 0)]);
    // d_minus_scale_output_by_2 = difference - 2*SqrtBB(x)
    let d_minus_scale_output_by_2 = graph.addNode(sub, vec![(difference, 0), (scale_output_by_2, 0)]);
    let _ = graph.addNode(non_negative_check, vec![(d_plus_scale_output_by_2, 0)]);
    let _ = graph.addNode(negative_check, vec![(d_minus_scale_output_by_2, 0)]);

    // Step 4. (X - mean) / sqrt(var + epsilon)
    let split_sub_output = graph.addNode(split_x, vec![(x_minus_mean_output, 0)]); // N*C outputs (shape: [1, D1, D2, ..., DN])
    let split_sqrt_output = graph.addNode(split_x, vec![(sqrt_output, 0)]); // N*C outputs (shape: [1, 1, 1, ..., 1])
    let split_scale_output = graph.addNode(split, vec![(scale_output, 0)]); // C outputs (shape: [1, 1, 1, ..., 1])
    let mut tmp_outputs = vec![];
    // here we perform the batch normalization for each channel
    let (N, C) = (mean_shape_padded[0], mean_shape_padded[1]);
    for n in 0..N as usize {
      for c in 0..C as usize {
        let idx = n * C as usize + c;
        let div_output = graph.addNode(div, vec![(split_sub_output, idx), (split_sqrt_output, idx)]);
        let a_SF2 = graph.addNode(mul_SF2, vec![(split_sub_output, idx)]);
        let a_SF2_plus_b = graph.addNode(add, vec![(a_SF2, 0), (split_sqrt_output, idx)]);
        let b2 = graph.addNode(mul_2, vec![(split_sqrt_output, idx)]);
        let qb2 = graph.addNode(mul_scalar, vec![(div_output, 0), (b2, 0)]);
        let qb2_plus_r = graph.addNode(add, vec![(qb2, 0), (div_output, 1)]);
        let _ = graph.addNode(eq, vec![(a_SF2_plus_b, 0), (qb2_plus_r, 0)]);
        // Now check r≥0:
        let _ = graph.addNode(range_check, vec![(div_output, 1)]);
        // Now check 2b-r≥0:
        let b2_minus_r = graph.addNode(sub, vec![(b2, 0), (div_output, 1)]);
        let _ = graph.addNode(range_check, vec![(b2_minus_r, 0)]);

        // Step 5. scale * (X - mean) / sqrt(var + epsilon)
        let mul_output = graph.addNode(mul_scalar, vec![(div_output, 0), (split_scale_output, c)]);
        let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
        let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);

        tmp_outputs.push((change_SF_output, 0));
      }
    }
    // concat_output (shape: [N*C, D1, D2, ..., DN])
    let concat_output = graph.addNode(concat, tmp_outputs);
    // reshape_concat (shape: [N, C, D1, D2, ..., DN])
    let concat_output = graph.addNode(reshape_concat, vec![(concat_output, 0)]);

    // Step 6. scale * (X - mean) / sqrt(var + epsilon) + bias
    let output = graph.addNode(add, vec![(concat_output, 0), (bias_output, 0)]);

    graph.outputs.push((output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}

// InstanceNorm is a struct that represents an instance normalization layer, which computes
// Y = (X - mean) * scale / sqrt(var + epsilon) + bias,
// where mean and var are computed per instance per channel
pub struct CustomInstanceNormLayer;
impl Layer for CustomInstanceNormLayer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    _constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>) {
    let mut graph = Graph::new();

    let X_shape = input_shapes[0];
    let scale_shape = input_shapes[1];
    let bias_shape = input_shapes[2];

    // Check that the shapes are correct
    // X: [N, D1*D2*...*DN, C]
    // scale: [C]
    // bias: [C]
    assert!(X_shape[2] == scale_shape[0] && scale_shape[0] == bias_shape[0]);
    assert!(scale_shape.len() == 1 && bias_shape.len() == 1);

    let epsilon_attr = attributes.iter().filter(|x| x.name == "epsilon").next();
    let mut epsilon = if let Some(x) = epsilon_attr {
      // epsilon is provided
      x.f as f32
    } else {
      // epsilon is not provided, use the default value
      1e-5
    };
    epsilon = 1.0;
    epsilon *= onnx::SF_FLOAT.read().unwrap().to_owned();

    // X_shape_for_mean: [N, C, D1 * D2 * ... * DN]
    let n = input_shapes[0].len();
    let (mut a, mut b) = (input_shapes[0][n - 2], input_shapes[0][n - 1]);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    let cc = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { n: a, m: b, permutation }),
      N: 2,
    }));
    let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
    let cc_back = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock {
        permutation: permutation_back,
        n: b,
        m: a,
      }),
      N: 2,
    }));

    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock { len: b }),
      N: 1,
    }));
    let div_const = graph.addBB(Box::new(DivConstProofBasicBlock { c: X_shape[1] as u32 }));
    let div_const_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: 1,
        setup: CQArrayType::NonNegative,
      }),
      N: 1,
    }));

    let mean_shape_padded = vec![util::next_pow(X_shape[0] as u32) as usize, util::next_pow(X_shape[2] as u32) as usize];
    //mean_shape_padded = mean_shape_padded.into_iter().chain(vec![1; X_shape.len() - 2]).collect();
    let reshape_0 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: mean_shape_padded.clone(),
    }));

    let (mut a, mut b) = (input_shapes[0][n - 1], 1);
    a = util::next_pow(a as u32) as usize;
    b = util::next_pow(b as u32) as usize;
    let permutation = ((0..b).map(|x| x * a).collect(), (0..a).collect());
    let cc2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { n: a, m: b, permutation }),
      N: 2,
    })); // to [1, C]
    let permutation_back = ((0..a).map(|x| x * b).collect(), (0..b).collect());
    let cc2_back = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock {
        permutation: permutation_back,
        n: b,
        m: a,
      }),
      N: 2,
    })); // to [C, 1]

    let epsilon = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(epsilon.round() as i32)]).into_dyn(),
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {
        len: util::next_pow(X_shape[1] as u32) as usize,
      }),
      N: 1,
    }));
    let sub = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SubBasicBlock {}),
      N: 1,
    }));
    let add = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(AddBasicBlock {}),
      N: 1,
    }));
    let sf_log = onnx::SF_LOG.read().unwrap().to_owned();
    let change_SF = graph.addBB(Box::new(ChangeSFBasicBlock {
      input_SF: sf_log * 2,
      output_SF: sf_log,
    }));
    let change_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: X_shape[X_shape.len() - 1].next_power_of_two(),
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
    let div_SF = graph.addBB(Box::new(DivConstBasicBlock {
      c: (vec![X_shape[1]].iter().fold(1, |x, &y| x * y) as f32).sqrt() * ((1 << sf_log) as f32),
    }));
    let div_SF_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        n: X_shape[X_shape.len() - 1].next_power_of_two(),
        setup: Some((
          Box::new(DivConstBasicBlock {
            c: (vec![X_shape[1]].iter().fold(1, |x, &y| x * y) as f32).sqrt() * ((1 << sf_log) as f32),
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
      N: 1,
    }));

    let sf = onnx::SF.read().unwrap().to_owned();
    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock { output_SF: sf }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: X_shape[1].next_power_of_two(),
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: sf * 2 }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
      N: 1,
    }));

    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: sf_log,
      output_SF: sf_log,
    }));
    let sf = onnx::SF.read().unwrap().to_owned();
    let sf_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(sf as i32)]).into_dyn(),
    }));
    let two_const = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(2)]).into_dyn(),
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let non_negative_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: util::next_pow(X_shape[1] as u32) as usize,
        setup: util::CQArrayType::NonNegative,
      }),
      N: 1,
    }));
    let non_positive_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: util::next_pow(X_shape[1] as u32) as usize,
        setup: util::CQArrayType::NonPositive,
      }),
      N: 1,
    }));

    let split_ind = vec![1; util::next_pow(scale_shape[0] as u32) as usize];
    let split_x = graph.addBB(Box::new(SplitBasicBlock { axis: 1, split: split_ind }));
    let concat = graph.addBB(Box::new(ConcatBasicBlock { axis: 1 }));
    let unsqueeze = graph.addBB(Box::new(UnsqueezeBasicBlock {}));

    // Step 0. Compute epsilon, mean, var, scale, and bias
    // output epsilon
    let epsilon_output = graph.addNode(epsilon, vec![]);

    // compute mean (shape: [N, C, 1, ..., 1])
    let cc_output = graph.addNode(cc, vec![(-1, 0)]); // [1, C, D1*D2*D3]
    let sum_output = graph.addNode(sum, vec![(cc_output, 0)]); // [1, C, 1]
    let mean_output = graph.addNode(div_const, vec![(sum_output, 0)]); // [1, C, 1]
    let _ = graph.addNode(div_const_check, vec![(mean_output, 1)]);
    let _ = graph.addNode(div_const_check, vec![(mean_output, 2)]);
    let mean_output = graph.addNode(cc2, vec![(mean_output, 0)]);
    let mean_output = graph.addNode(reshape_0, vec![(mean_output, 0)]); // [1, C]

    // compute var (shape: [N, C, 1, ..., 1])
    // m = mean(X)
    // s = size(X)
    // X - m
    let sub_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]); // [1, D1*D2*D3, C]
                                                                          // (X - m) * SF / (sqrt(s) * SF)
    let div_SF_output = graph.addNode(div_SF, vec![(sub_output, 0)]);
    let _ = graph.addNode(div_SF_check, vec![(sub_output, 0), (div_SF_output, 0)]);
    // (X - m)^2 / s
    let mul_output = graph.addNode(mul, vec![(sub_output, 0), (sub_output, 0)]);
    let cc_output = graph.addNode(cc, vec![(mul_output, 0)]); // [1, C, D1*D2*D3]
                                                              // SUM((X - m)^2) / s
    let var_output = graph.addNode(sum, vec![(cc_output, 0)]); // [1, C, 1]
    let var_output = graph.addNode(cc2, vec![(var_output, 0)]);
    let var_output = graph.addNode(reshape_0, vec![(var_output, 0)]); // [1, C]

    // Step 1. X - mean (shape: [N * C, D1, D2, ..., DN])
    let x_minus_mean_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]); // [1, D1*D2*D3, C]

    // Step 2. var + epsilon (shape: [N * C, 1, ..., 1])
    let var_plus_eps_output = graph.addNode(add, vec![(var_output, 0), (epsilon_output, 0)]); // [1, C]

    // Step 3. sqrt(var + epsilon) (shape: [N * C, 1, ..., 1])
    // x = var + epsilon
    // SqrtBB(x) = sqrt(x/SF)*SF + eps (where -1 < eps < 1)
    let sqrt_output = graph.addNode(sqrt, vec![(var_plus_eps_output, 0)]); // [1, C]
                                                                           // The following operations are to check if sqrt_output is correct
                                                                           // square_sqrt = SqrtBB(x)^2 = x*SF + 2*sqrt(x/SF)*SF*eps + eps^2
    let square_sqrt = graph.addNode(mul, vec![(sqrt_output, 0), (sqrt_output, 0)]);
    // scale_input_by_sf = x*SF
    let sf_const_output = graph.addNode(sf_const, vec![]);
    let scale_input_by_sf = graph.addNode(mul_scalar, vec![(var_plus_eps_output, 0), (sf_const_output, 0)]);
    // difference = SqrtBB(x)^2 - x*SF = 2*sqrt(x/SF)*SF*eps + eps^2 = 2*SqrtBB(x)*eps + eps^2
    // Because -1 < eps < 1, -2*SqrtBB(x) < 2*SqrtBB(x)*eps < 2*SqrtBB(x) and 0 < eps^2 < 1.

    // Therefore, - 2*SqrtBB(x) < difference < 2*SqrtBB(x) + 1.
    // The following two inequalities should hold:
    // 1. difference + 2*SqrtBB(x) >= 0
    // 2. difference - 2*SqrtBB(x) <= 0
    let difference = graph.addNode(sub, vec![(square_sqrt, 0), (scale_input_by_sf, 0)]);
    // scale_output_by_2 = 2*SqrtBB(x)
    let two_const_output = graph.addNode(two_const, vec![]);
    let scale_output_by_2 = graph.addNode(mul_scalar, vec![(sqrt_output, 0), (two_const_output, 0)]);
    let d_plus_scale_output_by_2 = graph.addNode(add, vec![(difference, 0), (scale_output_by_2, 0)]);
    // d_minus_scale_output_by_2 = difference - 2*SqrtBB(x)
    let d_minus_scale_output_by_2 = graph.addNode(sub, vec![(difference, 0), (scale_output_by_2, 0)]);
    let _ = graph.addNode(non_negative_check, vec![(d_plus_scale_output_by_2, 0)]);
    let _ = graph.addNode(non_positive_check, vec![(d_minus_scale_output_by_2, 0)]);

    let sqrt_output = graph.addNode(cc2_back, vec![(sqrt_output, 0)]); // [C, 1]
    let sqrt_output = graph.addNode(unsqueeze, vec![(sqrt_output, 0)]); // [1, C, 1]

    // Step 4. (X - mean) / sqrt(var + epsilon)
    let x_minus_mean_output = graph.addNode(cc, vec![(x_minus_mean_output, 0)]); // [1, C, D1*D2*D3]
    let split_sub_output = graph.addNode(split_x, vec![(x_minus_mean_output, 0)]); // C outputs (shape: [1, 1, D1*D2*D3])
    let split_sqrt_output = graph.addNode(split_x, vec![(sqrt_output, 0)]); // C outputs (shape: [1, 1, 1])
    let mut tmp_outputs = vec![];
    // here we perform the batch normalization for each channel
    let (N, C) = (mean_shape_padded[0], mean_shape_padded[1]);
    for n in 0..N as usize {
      for c in 0..C as usize {
        let idx = n * C as usize + c;
        let div_output = graph.addNode(div, vec![(split_sub_output, idx), (split_sqrt_output, idx)]);
        let a_SF2 = graph.addNode(mul_SF2, vec![(split_sub_output, idx)]);
        let a_SF2_plus_b = graph.addNode(add, vec![(a_SF2, 0), (split_sqrt_output, idx)]);
        let b2 = graph.addNode(mul_2, vec![(split_sqrt_output, idx)]);
        let qb2 = graph.addNode(mul_scalar, vec![(div_output, 0), (b2, 0)]);
        let qb2_plus_r = graph.addNode(add, vec![(qb2, 0), (div_output, 1)]);
        let _ = graph.addNode(eq, vec![(a_SF2_plus_b, 0), (qb2_plus_r, 0)]);
        // Now check r≥0:
        let _ = graph.addNode(range_check, vec![(div_output, 1)]);
        // Now check 2b-r≥0:
        let b2_minus_r = graph.addNode(sub, vec![(b2, 0), (div_output, 1)]);
        let _ = graph.addNode(range_check, vec![(b2_minus_r, 0)]);

        tmp_outputs.push((div_output, 0));
      }
    }
    // concat_output
    let concat_output = graph.addNode(concat, tmp_outputs); // [1, C, D1*D2*D3]
    let div_output = graph.addNode(cc_back, vec![(concat_output, 0)]); // [1, D1*D2*D3, C]

    // Step 5. scale * (X - mean) / sqrt(var + epsilon)
    let mul_output = graph.addNode(mul, vec![(div_output, 0), (-2, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);

    // Step 6. scale * (X - mean) / sqrt(var + epsilon) + bias
    let output = graph.addNode(add, vec![(change_SF_output, 0), (-3, 0)]);

    graph.outputs.push((output, 0));
    (graph, vec![input_shapes[0].clone()], vec![input_types[0]])
  }
}
