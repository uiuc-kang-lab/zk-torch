#[cfg(test)]
mod tests {
  use ark_bn254::{Fr, G1Affine, G2Affine};
  use ark_poly::univariate::DensePolynomial;
  use ndarray::ArrayD;
  use rand::{rngs::StdRng, SeedableRng};
  use rayon::prelude::*;
  use sha3::{Digest, Keccak256};
  use util::convert_to_data;
  use std::path::Path;
  use zk_torch::{basic_block::*, onnx, ptau, util};

  #[allow(non_snake_case)]
  fn test_layer(onnx_op_name: &str) {
    println!("Testing layer: {}", onnx_op_name);
    let srs = &ptau::load_file("challenge", 7, 7);
    let onnx_file_folder = "tests/ops/";
    let onnx_file_name = format!("{}.onnx", onnx_op_name);
    let onnx_file_name = Path::new(onnx_file_folder).join(onnx_file_name).to_str().unwrap().to_string();
    let (mut graph, models) = onnx::load_file(&onnx_file_name);
    let fake_inputs = util::generate_fake_inputs_for_onnx(&onnx_file_name);
    let inputs = fake_inputs.iter().map(|x| x).collect::<Vec<_>>();
    let models = models.iter().map(|x| x).collect::<Vec<_>>();

    // Run:
    let outputs = graph.run(&inputs, &models);

    // Setup:
    let models: Vec<ArrayD<Data>> = models.par_iter().map(|model| convert_to_data(srs, model)).collect();
    let models: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
    let setups = graph.setup(srs, &models);

    // Encode Data:
    let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> = util::vec_iter(&setups)
      .map(|x| {
        (
          util::vec_iter(&x.0).map(|y| (*y).into()).collect(),
          util::vec_iter(&x.1).map(|y| (*y).into()).collect(),
          util::vec_iter(&x.2).map(|y| (y.clone())).collect(),
        )
      })
      .collect();
    let setups = setups.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
    let modelsEnc: Vec<ArrayD<DataEnc>> = util::vec_iter(&models).map(|model| (*model).map(|x| DataEnc::new(srs, x))).collect();
    let inputs: Vec<ArrayD<Data>> = util::vec_iter(&inputs).map(|input| convert_to_data(srs, input)).collect();
    let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
    let inputsEnc: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
    let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
    let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();
    let outputs: Vec<Vec<ArrayD<Data>>> = graph.encodeOutputs(srs, &models, &inputs, &outputs);
    let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
    let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
    let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> =
      outputs.iter().map(|output| (*output).iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect()).collect();

    // Save files:
    let modelsEncBytes = bincode::serialize(&modelsEnc).unwrap();
    let inputsEncBytes = bincode::serialize(&inputsEnc).unwrap();
    let outputsEncBytes = bincode::serialize(&outputsEnc).unwrap();

    // Fiat-Shamir:
    let mut hasher = Keccak256::new();
    hasher.update(modelsEncBytes);
    hasher.update(inputsEncBytes);
    hasher.update(outputsEncBytes);
    let mut buf = [0u8; 32];
    hasher.finalize_into((&mut buf).into());
    let mut rng = StdRng::from_seed(buf);
    let mut rng2 = rng.clone();

    // Prove:
    let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);

    // Prepare for verification:
    let proofs = proofs.iter().map(|x| (&x.0, &x.1, &x.2)).collect();
    let modelsEnc: Vec<&ArrayD<DataEnc>> = modelsEnc.iter().map(|model| model).collect();
    let inputsEnc: Vec<&ArrayD<DataEnc>> = inputsEnc.iter().map(|input| input).collect();
    let outputsEnc: Vec<Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|output| output.iter().map(|x| x).collect()).collect();
    let outputsEnc: Vec<&Vec<&ArrayD<DataEnc>>> = outputsEnc.iter().map(|x| x).collect();

    // Verify:
    graph.verify(srs, &modelsEnc, &inputsEnc, &outputsEnc, &proofs, &mut rng2);
  }

  #[test]
  fn test_layers() {
    let supported_op = vec![
      "Add",
      "Mul",
      "Cast",
      "Ceil",
      "Concat",
      "ConstantOfShape",
      "Sub",
      "LSTM",
      "MatMul",
      "Relu",
      "Gather",
      "Range",
      "ReduceMean",
      "Pow",
      "Div",
      // "ScatterND", todo
      // "Slice",
      // "Split",
      // "Sqrt",
      // "Reshape",
      // "Transpose",
      // "Tanh",
      // "Shape",
      // "Sigmoid",
      // "Equal",
      // "Where",
      // "Expand",
      // "Softmax",
      // "Squeeze",
      // "Unsqueeze",
      // "Erf",
      // "Conv",
    ];
    for op in supported_op {
      test_layer(op);
    }
  }
}
