use crate::basic_block::*;
use crate::graph::*;
use crate::layer::Layer;
use crate::onnx;
use crate::util;
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
    epsilon *= *onnx::SF_FLOAT;

    let epsilon = graph.addBB(Box::new(Const2BasicBlock {
      c: arr1(&vec![Fr::from(epsilon.round() as i32)]).into_dyn(),
    }));
    let scale_shape_padded = util::next_pow(scale_shape[0] as u32) as usize;
    let reshape_1 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![1, scale_shape_padded],
    }));
    let permutation = ((0..scale_shape_padded).collect(), vec![0]);
    let permute = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
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
    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: *onnx::SF_LOG * 2,
      output_SF: *onnx::SF_LOG,
    }));
    let sqrt_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(SqrtBasicBlock {
            input_SF: *onnx::SF_LOG * 2,
            output_SF: *onnx::SF_LOG,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
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

    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock { output_SF: *onnx::SF }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: Array1::from_iter(0..*onnx::CQ_RANGE).map(|x| Fr::from(*x as i32)),
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: *onnx::SF * 2 }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
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
    let sqrt_output = graph.addNode(sqrt, vec![(add_output, 0)]);
    let _ = graph.addNode(sqrt_check, vec![(add_output, 0), (sqrt_output, 0)]);

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
    epsilon *= *onnx::SF_FLOAT;

    // X_shape_for_mean: [N, C, D1 * D2 * ... * DN]
    let x_shape_for_mean = vec![
      X_shape[0],
      X_shape[1],
      X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y),
    ];
    let permutation = get_reshape_indices(X_shape.to_vec(), x_shape_for_mean);
    let padded_input_shape: Vec<_> = X_shape.iter().map(|x| util::next_pow(*x as u32) as usize).collect();
    let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&padded_input_shape),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    }));

    let sum = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(SumBasicBlock {}),
      N: 1,
    }));
    let div_const = graph.addBB(Box::new(DivConstBasicBlock {
      c: X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y) as f32,
    }));
    let div_const_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(DivConstBasicBlock {
            c: X_shape.into_iter().skip(2).cloned().collect::<Vec<_>>().iter().fold(1, |x, &y| x * y) as f32,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
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
      basic_block: Box::new(PermuteBasicBlock { permutation: permutation }),
      N: 2,
    }));
    let num_one = X_shape.len() - 2;
    let reshape_2 = graph.addBB(Box::new(ReshapeBasicBlock {
      shape: vec![scale_shape_padded].into_iter().chain(vec![1; num_one]).collect(),
    }));
    let mul = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulBasicBlock {}),
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
    let sqrt = graph.addBB(Box::new(SqrtBasicBlock {
      input_SF: *onnx::SF_LOG * 2,
      output_SF: *onnx::SF_LOG,
    }));
    let sqrt_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQ2BasicBlock {
        setup: Some((
          Box::new(SqrtBasicBlock {
            input_SF: *onnx::SF_LOG * 2,
            output_SF: *onnx::SF_LOG,
          }),
          *onnx::CQ_RANGE_LOWER,
          *onnx::CQ_RANGE,
        )),
      }),
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

    let div = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(DivScalarBasicBlock { output_SF: *onnx::SF }),
      N: 1,
    }));
    let range_check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        setup: Array1::from_iter(0..*onnx::CQ_RANGE).map(|x| Fr::from(*x as i32)),
      }),
      N: 1,
    }));
    let mul_SF2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: *onnx::SF * 2 }),
      N: 1,
    }));
    let mul_2 = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulConstBasicBlock { c: 2 }),
      N: 1,
    }));
    let mul_scalar = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(MulScalarBasicBlock {}),
      N: 1,
    }));
    let eq = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(EqBasicBlock {}),
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
    let _ = graph.addNode(div_const_check, vec![(sum_output, 0), (mean_output, 0)]);
    let mean_output = graph.addNode(reshape_0, vec![(mean_output, 0)]);

    // compute var (shape: [N, C, 1, ..., 1])
    let sub_output = graph.addNode(sub, vec![(-1, 0), (mean_output, 0)]);
    let mul_output = graph.addNode(mul, vec![(sub_output, 0), (sub_output, 0)]);
    let change_SF_output = graph.addNode(change_SF, vec![(mul_output, 0)]);
    let _ = graph.addNode(change_SF_check, vec![(mul_output, 0), (change_SF_output, 0)]);
    let cc_output = graph.addNode(cc, vec![(change_SF_output, 0)]);
    let sum_output = graph.addNode(sum, vec![(cc_output, 0)]);
    let var_output = graph.addNode(div_const, vec![(sum_output, 0)]);
    let _ = graph.addNode(div_const_check, vec![(sum_output, 0), (var_output, 0)]);
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
    let sqrt_output = graph.addNode(sqrt, vec![(var_plus_eps_output, 0)]);
    let _ = graph.addNode(sqrt_check, vec![(var_plus_eps_output, 0), (sqrt_output, 0)]);

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
