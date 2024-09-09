use crate::basic_block::*;
use crate::{ptau, util, util::convert_to_data};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::univariate::DensePolynomial;
use ark_std::UniformRand;
use ark_std::{One, Zero};
use ndarray::{arr0, concatenate, s, ArrayD, Axis, IxDyn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn testBasicBlock<BB: BasicBlock>(basic_block: BB, srs: &SRS, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) {
  let mut rng = StdRng::from_entropy();
  let outputs = basic_block.run(model, inputs);
  let outputs: Vec<&ArrayD<Fr>> = outputs.iter().map(|x| x).collect();
  let model = convert_to_data(srs, model);
  let setup = basic_block.setup(srs, &model);
  let setup: (Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>) = (
    setup.0.iter().map(|y| (*y).into()).collect(),
    setup.1.iter().map(|y| (*y).into()).collect(),
    setup.2.iter().map(|y| (y.clone())).collect(),
  );
  let setup = (&setup.0, &setup.1, &setup.2);
  let inputs: Vec<ArrayD<Data>> = inputs.iter().map(|input| convert_to_data(srs, input)).collect();
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<Data>> = basic_block.encodeOutputs(srs, &model, &inputs, &outputs);
  let outputs: Vec<&ArrayD<Data>> = outputs.iter().map(|output| output).collect();
  let mut rng2 = rng.clone();
  let cache = Arc::new(Mutex::new(HashMap::new()));
  let proof = basic_block.prove(srs, setup, &model, &inputs, &outputs, &mut rng, cache.clone());
  let proof: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (
    proof.0.iter().map(|y| (*y).into()).collect(),
    proof.1.iter().map(|y| (*y).into()).collect(),
    proof.2.iter().map(|y| (*y)).collect(),
  );
  let proof = (&proof.0, &proof.1, &proof.2);
  let model = model.map(|x| DataEnc::new(srs, x));
  let inputs: Vec<ArrayD<DataEnc>> = inputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let inputs: Vec<&ArrayD<DataEnc>> = inputs.iter().map(|input| input).collect();
  let outputs: Vec<ArrayD<DataEnc>> = outputs.iter().map(|x| (*x).map(|y| DataEnc::new(srs, y))).collect();
  let outputs: Vec<&ArrayD<DataEnc>> = outputs.iter().map(|output| output).collect();
  let cache = Arc::new(Mutex::new(HashMap::new()));
  let pairings = basic_block.verify(srs, &model, &inputs, &outputs, proof, &mut rng2, cache.clone());
  let pairings = pairings.iter().map(|x| x).collect();
  let pairings = util::combine_pairing_checks(&pairings);
  assert_eq!(Bn254::multi_pairing(pairings.0.iter(), pairings.1.iter()), PairingOutput::zero());
  // Check that prove and verify end with the same rng state
  assert_eq!(Fr::rand(&mut rng), Fr::rand(&mut rng2));
}

#[test]
fn testBasicBlocks() {
  let srs = &ptau::load_file("challenge", 7, 7);
  let mut rng = StdRng::from_entropy();
  let N: usize = 1 << 6;
  let n: usize = 1 << 3;
  let a = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let a_n = a.slice(s![..n]).to_owned().into_dyn();
  let a_0 = arr0(a[0]).into_dyn();
  let b = ArrayD::from_shape_fn(IxDyn(&[N]), |_| Fr::rand(&mut rng));
  let b_n = b.slice(s![..n]).to_owned().into_dyn();
  let temp1 = a.view().into_shape(IxDyn(&[1, N])).unwrap();
  let temp2 = b.view().into_shape(IxDyn(&[1, N])).unwrap();
  let ab = concatenate(Axis(0), &[temp1, temp2]).unwrap();
  let empty = ArrayD::zeros(IxDyn(&[0]));
  testBasicBlock(EqBasicBlock {}, srs, &empty, &vec![&a, &a]);
  testBasicBlock(AddBasicBlock {}, srs, &empty, &vec![&a, &b]);
  testBasicBlock(SubBasicBlock {}, srs, &empty, &vec![&a, &b]);
  testBasicBlock(MulBasicBlock {}, srs, &empty, &vec![&a, &b]);
  testBasicBlock(MulConstBasicBlock { c: 12345 }, srs, &empty, &vec![&a]);
  testBasicBlock(MulScalarBasicBlock {}, srs, &empty, &vec![&a, &a_0]);
  testBasicBlock(AddBasicBlock {}, srs, &empty, &vec![&a_0, &b]);
  testBasicBlock(AddBasicBlock {}, srs, &empty, &vec![&b, &a_0]);
  testBasicBlock(SubBasicBlock {}, srs, &empty, &vec![&a_0, &b]);
  testBasicBlock(SubBasicBlock {}, srs, &empty, &vec![&b, &a_0]);
  testBasicBlock(SubBasicBlock {}, srs, &empty, &vec![&a_0, &a_0]);
  testBasicBlock(
    CQBasicBlock {
      setup: a.clone().into_dimensionality::<ndarray::Ix1>().unwrap(),
    },
    srs,
    &a,
    &vec![&a_n],
  );
  testBasicBlock(CQ2BasicBlock { setup: None }, srs, &ab, &vec![&a_n, &b_n]);
  testBasicBlock(SumBasicBlock {}, srs, &empty, &vec![&a]);

  let data_to_split = ArrayD::from_shape_fn(IxDyn(&[4, 2]), |_| Fr::rand(&mut rng));
  testBasicBlock(
    SplitBasicBlock {
      axis: 0,
      split: vec![1, 1, 1, 1],
    },
    srs,
    &empty,
    &vec![&data_to_split],
  );
  let data_to_concat = ArrayD::from_shape_fn(IxDyn(&[1, 2]), |_| Fr::rand(&mut rng));
  testBasicBlock(ConcatBasicBlock { axis: 0 }, srs, &empty, &vec![&data_to_concat, &data_to_concat.clone()]);

  let l: usize = 1 << 3;
  let m: usize = 1 << 2;
  let n: usize = 1 << 1;
  let a = ArrayD::from_shape_fn(IxDyn(&[l, m]), |_| Fr::rand(&mut rng));
  let b = ArrayD::from_shape_fn(IxDyn(&[n, m]), |_| Fr::rand(&mut rng));
  let c = ArrayD::from_shape_fn(IxDyn(&[m, n]), |_| Fr::rand(&mut rng));
  testBasicBlock(MatMulBasicBlock {}, srs, &empty, &vec![&a, &b]);
  testBasicBlock(CQLinBasicBlock {}, srs, &c, &vec![&a]);
  let p1 = (vec![0], (0..l * m).collect::<Vec<_>>()); // Concatenate columns
  let p2 = (vec![0], (0..l * m).map(|i| (i % m) * l + (i / m)).collect::<Vec<_>>()); // Concatenate rows
  let p3 = ((0..m).map(|i| i * l).collect::<Vec<_>>(), (0..l).collect::<Vec<_>>()); // Transpose
  testBasicBlock(PermuteBasicBlock { permutation: p1 }, srs, &empty, &vec![&a]);
  testBasicBlock(PermuteBasicBlock { permutation: p2 }, srs, &empty, &vec![&a]);
  testBasicBlock(PermuteBasicBlock { permutation: p3 }, srs, &empty, &vec![&a]);
  let min = 1.;
  let max = 8.;
  testBasicBlock(ClipBasicBlock { min, max }, srs, &empty, &vec![&a]);

  // generate booleans
  let a = ArrayD::from_shape_fn(IxDyn(&[4]), |_| Fr::from(rng.gen_range(0..2)));
  testBasicBlock(BooleanCheckBasicBlock {}, srs, &empty, &vec![&a]);

  // test TopK
  let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![3, 4, 7, 1].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  let a_idx = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0, 1, 2, 3].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  testBasicBlock(SortBasicBlock { descending: true, len: 4 }, srs, &empty, &vec![&a, &a_idx]);
  let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1, 2, 3, 4].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  let a_ind = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0, 1, 2, 3].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![4, 3, 2, 1].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  let b_ind = ArrayD::from_shape_vec(IxDyn(&[4]), vec![3, 2, 1, 0].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  testBasicBlock(OneToOneBasicBlock {}, srs, &empty, &vec![&a, &a_ind, &b, &b_ind]);
  testBasicBlock(OrderedBasicBlock {}, srs, &empty, &vec![&b]);
}

#[test]
fn test_max() {
  let CQ_RANGE_LOWER: i32 = -(1 << 5);
  let srs = &ptau::load_file("challenge", 7, 7);
  let empty = ArrayD::zeros(IxDyn(&[0]));
  let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1, 0].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  testBasicBlock(
    MaxProofBasicBlock {
      cq_range_lower: CQ_RANGE_LOWER,
    },
    srs,
    &empty,
    &vec![&a],
  );
  let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2, 2, -1, 4].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  testBasicBlock(
    MaxProofBasicBlock {
      cq_range_lower: CQ_RANGE_LOWER,
    },
    srs,
    &empty,
    &vec![&b],
  );
  let c = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-4, -3, -1, -2].into_iter().map(|x| Fr::from(x)).collect()).unwrap();
  testBasicBlock(
    MaxProofBasicBlock {
      cq_range_lower: CQ_RANGE_LOWER,
    },
    srs,
    &empty,
    &vec![&c],
  );
}

#[test]
fn test_copy_constraint() {
  let srs = &ptau::load_file("challenge", 7, 7);
  let empty = ArrayD::zeros(IxDyn(&[0]));
  let permutation = ArrayD::from_shape_vec(vec![4], vec![Some(IxDyn(&[3])), Some(IxDyn(&[2])), Some(IxDyn(&[1])), Some(IxDyn(&[0]))]).unwrap();
  // reverse
  testBasicBlock(
    CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&[4]),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    },
    srs,
    &empty,
    &vec![&ArrayD::from_shape_vec(vec![4], (1..5).map(|x| Fr::from(x)).collect()).unwrap()],
  );
  // transpose
  let permutation = ArrayD::from_shape_vec(
    vec![2, 2],
    vec![Some(IxDyn(&[1, 1])), Some(IxDyn(&[1, 0])), Some(IxDyn(&[0, 1])), Some(IxDyn(&[0, 0]))],
  )
  .unwrap();
  testBasicBlock(
    CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&[2, 2]),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    },
    srs,
    &empty,
    &vec![&ArrayD::from_shape_vec(vec![2, 2], (1..5).map(|x| Fr::from(x)).collect()).unwrap()],
  );
  // 2d -> 3d
  let permutation = ArrayD::from_shape_vec(
    vec![2, 2, 4],
    vec![
      Some(IxDyn(&[1, 1])),
      Some(IxDyn(&[2, 0])),
      Some(IxDyn(&[3, 1])),
      Some(IxDyn(&[0, 0])),
      Some(IxDyn(&[1, 1])),
      Some(IxDyn(&[2, 0])),
      Some(IxDyn(&[3, 1])),
      Some(IxDyn(&[0, 0])),
      Some(IxDyn(&[1, 1])),
      Some(IxDyn(&[2, 0])),
      Some(IxDyn(&[3, 1])),
      Some(IxDyn(&[0, 0])),
      Some(IxDyn(&[1, 1])),
      Some(IxDyn(&[2, 0])),
      Some(IxDyn(&[3, 1])),
      None,
    ],
  )
  .unwrap();
  testBasicBlock(
    CopyConstraintBasicBlock {
      permutation,
      input_dim: IxDyn(&[4, 2]),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    },
    srs,
    &empty,
    &vec![&ArrayD::from_shape_vec(vec![4, 2], (1..9).map(|x| Fr::from(x)).collect()).unwrap()],
  );
  // 3d -> 2d with padding
  testBasicBlock(
    CopyConstraintBasicBlock {
      permutation: ArrayD::from_shape_vec(
        vec![4, 4],
        vec![
          Some(IxDyn(&[1, 1, 0])),
          Some(IxDyn(&[1, 0, 1])),
          Some(IxDyn(&[0, 1, 0])),
          None,
          Some(IxDyn(&[0, 0, 0])),
          Some(IxDyn(&[0, 1, 0])),
          Some(IxDyn(&[1, 1, 0])),
          None,
          Some(IxDyn(&[0, 1, 1])),
          Some(IxDyn(&[0, 0, 0])),
          Some(IxDyn(&[1, 0, 1])),
          None,
          None,
          None,
          None,
          None,
        ],
      )
      .unwrap(),
      input_dim: IxDyn(&[2, 2, 4]),
      padding_partition: copy_constraint::PaddingEnum::Max(Fr::one()),
    },
    srs,
    &empty,
    &vec![&ArrayD::from_shape_vec(vec![2, 2, 4], (1..17).map(|x| Fr::from(x)).collect()).unwrap()],
  );
  // slice
  testBasicBlock(
    CopyConstraintBasicBlock {
      permutation: ArrayD::from_shape_vec(
        vec![1, 1, 4],
        vec![
          Some(IxDyn(&[0, 0, 0])),
          Some(IxDyn(&[0, 0, 1])),
          Some(IxDyn(&[0, 0, 2])),
          Some(IxDyn(&[0, 0, 3])),
        ],
      )
      .unwrap(),
      input_dim: IxDyn(&[2, 1, 4]),
      padding_partition: copy_constraint::PaddingEnum::Zero,
    },
    srs,
    &empty,
    &vec![&ArrayD::from_shape_vec(vec![2, 1, 4], (1..9).map(|x| Fr::from(x)).collect()).unwrap()],
  );
}
