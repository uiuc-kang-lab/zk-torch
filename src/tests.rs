use crate::{basic_block::*, convert_to_data, graph::SetupType, ptau, util};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::{UniformRand, Zero};
use ndarray::{arr0, concatenate, s, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

fn testBasicBlock<BB: BasicBlock>(mut basic_block: BB, srs: &SRS, setup: &SetupType, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let outputs = basic_block.run(model, inputs);
  let outputs: Vec<&ArrayD<Fr>> = outputs.iter().map(|x| x).collect();
  let mut model = &convert_to_data(srs, model);
  model = if let SetupType::CQLin(s) = setup {
    &s.weights
  } else if let SetupType::CQ(s) = setup {
    &s.table
  } else {
    model
  };

  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<Data>> = basic_block.encodeOutputs(srs, &inputs, &outputs);
  let outputs: Vec<&ArrayD<Data>> = outputs.iter().map(|output| output).collect();
  let mut rng2 = rng.clone();
  let proof = basic_block.prove(srs, setup, &inputs, &outputs, &mut rng);
  let proof: (Vec<G1Affine>, Vec<G2Affine>) = (
    proof.0.iter().map(|y| (*y).into()).collect(),
    proof.1.iter().map(|y| (*y).into()).collect(),
  );
  let proof = (&proof.0, &proof.1);

  let model = model.map(|x| DataEnc::new(srs, x));
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<DataEnc>> = outputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<&ArrayD<DataEnc>> = outputs.iter().map(|output| output).collect();
  let pairings = basic_block.verify(srs, &model, &inputs, &outputs, proof, &mut rng2);
  let pairings = pairings.iter().map(|x| x).collect();
  let pairings = util::combine_pairing_checks(&pairings);
  assert_eq!(Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()), PairingOutput::zero());
}

#[test]
fn testBasicBlocks() {
  let srs = &ptau::load_file("challenge", 7);
  let mut rng = StdRng::from_entropy();
  let N: usize = 1 << 6;
  let n: usize = 1 << 3;
  let a = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let a_n = a.slice(s![..n]).to_owned().into_dyn();
  let a_1 = arr0(a[0]).into_dyn();
  let b = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let b_n = b.slice(s![..n]).to_owned().into_dyn();
  let temp1 = a.view().into_shape(IxDyn(&[1, N])).unwrap();
  let temp2 = b.view().into_shape(IxDyn(&[1, N])).unwrap();
  let ab = concatenate(Axis(0), &[temp1, temp2]).unwrap();
  let empty = ArrayD::zeros(IxDyn(&[0]));
  let setup0 = SetupType::None;
  testBasicBlock(EqBasicBlock {}, srs, &setup0, &empty, &vec![&a, &a]);
  testBasicBlock(AddBasicBlock {}, srs, &setup0, &empty, &vec![&a, &b]);
  testBasicBlock(SubBasicBlock {}, srs, &setup0, &empty, &vec![&a, &b]);
  testBasicBlock(MulBasicBlock {}, srs, &setup0, &empty, &vec![&a, &b]);
  testBasicBlock(MulConstBasicBlock { c: 12345 }, srs, &setup0, &empty, &vec![&a]);
  testBasicBlock(MulScalarBasicBlock {}, srs, &setup0, &empty, &vec![&a, &a_1]);
  testBasicBlock(AddBasicBlock {}, srs, &setup0, &empty, &vec![&a_1, &b]);
  testBasicBlock(AddBasicBlock {}, srs, &setup0, &empty, &vec![&b, &a_1]);
  testBasicBlock(SubBasicBlock {}, srs, &setup0, &empty, &vec![&a_1, &b]);
  testBasicBlock(SubBasicBlock {}, srs, &setup0, &empty, &vec![&b, &a_1]);

  let cq_bb = CQBasicBlock {
    table_dict: HashMap::new(),
    name: "test1".to_string(),
  };
  let setup1 = cq_bb.setup(&srs, &a);

  let l: usize = 1 << 3;
  let m: usize = 1 << 2;
  let n: usize = 1 << 1;
  let c = ArrayD::from_shape_fn(IxDyn(&[m, n]), |_| Fr::rand(&mut rng));
  let weights_map = HashMap::from([("test3".to_string(), c.clone())]);
  testBasicBlock(cq_bb, srs, &setup1, &a, &vec![&a_n]);

  let cq2_bb = CQ2BasicBlock {
    table_dict: HashMap::new(),
    name: "test2".to_string(),
  };
  let setup2 = cq2_bb.setup(&srs, &ab);
  testBasicBlock(cq2_bb, srs, &setup2, &ab, &vec![&a_n, &b_n]);

  let a = ArrayD::from_shape_fn(IxDyn(&[l, m]), |_| Fr::rand(&mut rng));
  let b = ArrayD::from_shape_fn(IxDyn(&[n, m]), |_| Fr::rand(&mut rng));
  testBasicBlock(MatMulBasicBlock {}, srs, &setup0, &empty, &vec![&a, &b]);
  testBasicBlock(SumBasicBlock {}, srs, &setup0, &empty, &vec![&a]);

  let cqlin_bb = CQLinBasicBlock {
    weights_name: "test3".to_string(),
  };
  let weight_setups: HashMap<_, _> = weights_map.iter().map(|(k, v)| (k.clone(), cqlin_bb.setup(&srs, v))).collect();
  let setup3 = weight_setups.get(&"test3".to_string()).unwrap();
  testBasicBlock(cqlin_bb, srs, &setup3, &c, &vec![&a]);
  let p1 = (vec![0], (0..l * m).collect::<Vec<_>>()); // Concatenate columns
  let p2 = (vec![0], (0..l * m).map(|i| (i % m) * l + (i / m)).collect::<Vec<_>>()); // Concatenate rows
  let p3 = ((0..m).map(|i| i * l).collect::<Vec<_>>(), (0..l).collect::<Vec<_>>()); // Transpose
  testBasicBlock(PermuteBasicBlock { permutation: p1 }, srs, &setup0, &empty, &vec![&a]);
  testBasicBlock(PermuteBasicBlock { permutation: p2 }, srs, &setup0, &empty, &vec![&a]);
  testBasicBlock(PermuteBasicBlock { permutation: p3 }, srs, &setup0, &empty, &vec![&a]);
}
