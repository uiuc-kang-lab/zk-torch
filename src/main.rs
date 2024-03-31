#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::Fr;
use ark_bn254::{G1Affine, G2Affine};
use basic_block::*;
use graph::Graph;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
mod basic_block;
mod graph;
mod onnx_converter;
mod ptau;
#[cfg(test)]
mod tests;
mod util;

fn main() {
  let srs = &ptau::load_file("challenge", 7);
  let (mut graph, updated_models) = Graph::build_from_onnx("network.onnx").unwrap();

  // make models from Vec<Vec<Vec<Fr>>> to Vec<&Vec<&Vec<Fr>>
  let mut models_ref = vec![vec![]; updated_models.len()];
  for (i, mm) in updated_models.iter().enumerate() {
    for (_, mmm) in mm.iter().enumerate() {
      models_ref[i].push(mmm);
    }
  }
  let mut models = vec![];
  for (_, mm) in models_ref.iter().enumerate() {
    models.push(mm);
  }

  // create fake input tensor
  let input: Vec<_> = (0..2).into_par_iter().map_init(rand::thread_rng, |rng, _| Fr::from(rng.gen_range(-4..4))).collect();

  //Run:
  let inputs = vec![&input];
  let outputs = graph.run(&inputs, &models);

  //Setup:
  let models: Vec<Vec<_>> = models.iter().map(|model| model.iter().map(|x| Data::new(srs, x)).collect()).collect();
  let models: Vec<_> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models = models.iter().map(|x| x).collect();
  let setups = graph.setup(srs, &models);
  //Converting to affine
  let setups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    setups.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let setups = setups.iter().map(|x| (&x.0, &x.1)).collect();

  //Prove:
  let inputs: Vec<_> = inputs.iter().map(|x| Data::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| Data::new(srs, x)).collect()).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| x).collect()).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| x).collect();
  let mut rng = StdRng::from_entropy();
  let mut rng2 = rng.clone();
  let proofs = graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng);
  //Converting to affine
  let proofs: Vec<(Vec<G1Affine>, Vec<G2Affine>)> =
    proofs.iter().map(|x| (x.0.iter().map(|y| (*y).into()).collect(), x.1.iter().map(|y| (*y).into()).collect())).collect();
  let proofs = proofs.iter().map(|x| (&x.0, &x.1)).collect();

  //Verify:
  let models: Vec<Vec<_>> = models.iter().map(|model| (**model).iter().map(|x| DataEnc::new(srs, *x)).collect()).collect();
  let models: Vec<_> = models.iter().map(|model| model.iter().map(|x| x).collect()).collect();
  let models = models.iter().map(|x| x).collect();
  let inputs: Vec<_> = inputs.iter().map(|x| DataEnc::new(srs, x)).collect();
  let inputs = inputs.iter().map(|x| x).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| DataEnc::new(srs, x)).collect()).collect();
  let outputs: Vec<Vec<_>> = outputs.iter().map(|x| x.iter().map(|x| x).collect()).collect();
  let outputs: Vec<_> = outputs.iter().map(|x| x).collect();
  graph.verify(srs, &models, &inputs, &outputs, &proofs, &mut rng2);
}
