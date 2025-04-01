use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::basic_block::*;
use crate::{
  ndarr_azip,
  util::{self, acc_proof_to_acc, acc_to_acc_proof, AccHolder, AccProofLayout},
};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::univariate::DensePolynomial;
use itertools::multiunzip;
use ndarray::{arr1, azip, par_azip, s, ArrayD, Axis, Dimension, IxDyn, SliceInfo, SliceInfoElem};
use rand::rngs::StdRng;
use rayon::prelude::*;

macro_rules! downcast_to_layout {
  ($bb:expr, $( $ty:ty ),+ ) => {
    {
      let bb_ref: &dyn AccProofLayout =
        $(
          if $bb.is::<$ty>() {
            $bb.downcast_ref::<$ty>().unwrap() as &dyn AccProofLayout
          } else
        )+
      {
        &BasicBlockForTest {} as &dyn AccProofLayout
      };
      bb_ref
    }
  };
}

fn get_local_acc_proof_indices(
  bb: &dyn BasicBlock,
  acc_g1_len: usize,
  acc_fr_len: usize,
  is_prover: bool,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
  if acc_fr_len == 0 {
    return (vec![0, 0], vec![0, 0], vec![0, 0], vec![0, 0]);
  }
  let bb: &dyn AccProofLayout = downcast_to_layout!(
    bb,
    MulBasicBlock,
    MulScalarBasicBlock,
    MulConstBasicBlock,
    SumBasicBlock,
    CQLinBasicBlock,
    MatMulBasicBlock,
    PermuteBasicBlock,
    CQBasicBlock,
    CQ2BasicBlock
  );
  let local_bb_num = acc_fr_len / (bb.acc_fr_num(is_prover) + 2 * bb.err_fr_nums().iter().sum::<usize>() + 1);
  let mut acc_g1_indices = vec![0];
  let mut acc_g2_indices = vec![0];
  let mut acc_fr_indices = vec![0];
  let mut acc_gt_indices = vec![0];

  let base_g1 = bb.acc_g1_num(is_prover) + 2 * (bb.err_g1_nums().iter().sum::<usize>());
  let base_g2 = bb.acc_g2_num(is_prover) + 2 * (bb.err_g2_nums().iter().sum::<usize>());
  let base_fr = bb.acc_fr_num(is_prover) + 2 * bb.err_fr_nums().iter().sum::<usize>() + 1;
  let base_gt = 2 * bb.err_gt_nums().iter().sum::<usize>();

  for i in 0..local_bb_num {
    acc_g1_indices.push(acc_g1_indices.last().unwrap() + base_g1);
    acc_g2_indices.push(acc_g2_indices.last().unwrap() + base_g2);
    acc_fr_indices.push(acc_fr_indices.last().unwrap() + base_fr);
    acc_gt_indices.push(acc_gt_indices.last().unwrap() + base_gt);
  }
  (acc_g1_indices, acc_g2_indices, acc_fr_indices, acc_gt_indices)
}

#[derive(Debug)]
pub struct RepeaterBasicBlock {
  pub basic_block: Box<dyn BasicBlock>,
  pub N: usize,
}

fn broadcastN<T1: Clone + std::fmt::Debug, T2: Clone + std::fmt::Debug>(
  inputs: &Vec<&ArrayD<T1>>,
  outputs: Option<&Vec<&ArrayD<T2>>>,
  N: usize,
) -> ArrayD<(Vec<ArrayD<T1>>, Option<Vec<ArrayD<T2>>>)> {
  // Broadcast inputs to a shared larger dimension
  let dims: Vec<_> = inputs.iter().map(|input| input.shape().to_vec()).collect();
  let dims_ptr = dims.iter().map(|x| x).collect();
  let superDim = util::broadcastDims(&dims_ptr, N);
  let len = superDim.len() + N;
  let broadcasted: Vec<_> = inputs
    .iter()
    .zip(dims)
    .map(|(input, dim)| {
      let mut newDim = superDim.clone();
      if dim.len() < N {
        newDim.extend_from_slice(&dim[..]);
      } else {
        newDim.extend_from_slice(&dim[dim.len() - N..]);
      }
      input.broadcast(newDim).unwrap()
    })
    .collect();

  // Run basic_block on last "N" dimensions, and combine the results
  // Assumes basic_block always has outputs of the same shape
  let sliceAll = SliceInfoElem::Slice {
    start: 0,
    end: None,
    step: 1,
  };
  ArrayD::from_shape_fn(superDim.clone(), |idx| {
    let idx = idx.slice().to_vec();
    let mut slice: Vec<_> = idx.iter().map(|x| SliceInfoElem::Index(*x as isize)).collect();
    slice.resize(len, sliceAll);
    let localInputs: Vec<_> = broadcasted
      .iter()
      .map(|arr| {
        let sliceInfo: SliceInfo<_, IxDyn, IxDyn> = SliceInfo::try_from(&slice[..arr.ndim()]).unwrap();
        arr.slice(sliceInfo).to_owned()
      })
      .collect();
    let localOutputs = match outputs {
      None => None,
      Some(outputs) => Some(
        outputs
          .iter()
          .map(|output| {
            let mut slice: Vec<_> = idx.iter().map(|x| SliceInfoElem::Index(*x as isize)).collect();
            slice.resize(output.ndim(), sliceAll);
            let sliceInfo: SliceInfo<_, IxDyn, IxDyn> = SliceInfo::try_from(&slice[..]).unwrap();
            output.slice(sliceInfo).to_owned()
          })
          .collect(),
      ),
    };
    (localInputs, localOutputs)
  })
}

fn combineArr<T: Clone>(arr: &ArrayD<&Vec<&ArrayD<T>>>) -> Vec<ArrayD<T>> {
  let mut outputs = None;
  let mut outputDims = None;
  arr.for_each(|localOutputs| match outputs.as_mut() {
    None => {
      outputs = Some(localOutputs.iter().map(|x| x.as_slice().unwrap().to_vec()).collect::<Vec<_>>());
      outputDims = Some(localOutputs.iter().map(|x| x.shape().to_vec()).collect::<Vec<_>>());
    }
    Some(outputs) => localOutputs.iter().enumerate().for_each(|(i, x)| outputs[i].extend_from_slice(x.as_slice().unwrap())),
  });
  let outputs: Vec<_> = outputs
    .unwrap()
    .into_iter()
    .zip(outputDims.unwrap())
    .map(|(output, outputDim)| {
      let mut newOutputDim = arr.shape().to_vec();
      newOutputDim.extend_from_slice(&outputDim);
      ArrayD::from_shape_vec(newOutputDim, output).unwrap()
    })
    .collect();
  outputs
}

impl BasicBlock for RepeaterBasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    self.basic_block.genModel()
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let temp = broadcastN::<Fr, Fr>(inputs, None, self.N);
    let temp = temp.map(|(subArrays, _)| {
      let subArrays: Vec<_> = util::vec_iter(subArrays).map(|y| y).collect();
      self.basic_block.run(model, &subArrays)
    });
    if temp.iter().any(|x| x.is_err()) {
      return Err(util::CQOutOfRangeError {
        input: temp.iter().filter_map(|x| x.as_ref().err()).next().unwrap().input,
      });
    }
    let temp = temp.map(|x| x.as_ref().unwrap());
    let temp = temp.map(|x| x.iter().map(|y| y).collect());
    let temp = temp.map(|x| x);
    Ok(combineArr(&temp))
  }

  fn encodeOutputs(&self, srs: &SRS, model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let mut temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let mut empty = ArrayD::from_elem(temp.shape(), vec![]);
    ndarr_azip!(((localInputs, localOutputs) in &mut temp, x in &mut empty) {
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      *x = self.basic_block.encodeOutputs(srs, model, &localInputs, &localOutputs);
    });
    let temp = empty.map(|x| x.iter().map(|y| y).collect());
    let temp = temp.map(|x| x);
    combineArr(&temp)
  }

  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    self.basic_block.setup(srs, model)
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let mut empty = ArrayD::from_elem(temp.shape(), (vec![], vec![], vec![]));
    ndarr_azip!(((localInputs, localOutputs) in &mut temp, x in &mut empty) {
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      let mut rng = rng.clone();
      let tmp = self.basic_block.prove(srs, setup, model, &localInputs, &localOutputs, &mut rng, cache.clone());
      *x = tmp;
    });
    let proof: (Vec<_>, Vec<_>, Vec<_>) = multiunzip(empty.into_iter());
    let proof = (
      proof.0.into_iter().flatten().collect(),
      proof.1.into_iter().flatten().collect(),
      proof.2.into_iter().flatten().collect(),
    );
    proof
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut temp = broadcastN(inputs, Some(outputs), self.N - 1);

    let l = temp.len();
    let divA = proof.0.len() / l;
    let divB = proof.1.len() / l;
    let divC = proof.2.len() / l;
    let combined: Vec<_> = (0..l)
      .map(|i| {
        (
          &proof.0[i * divA..i * divA + divA],
          &proof.1[i * divB..i * divB + divB],
          &proof.2[i * divC..i * divC + divC],
        )
      })
      .collect();
    let mut proofArr = ArrayD::from_shape_vec(temp.shape(), combined).unwrap();

    let mut empty = ArrayD::from_elem(temp.shape(), vec![]);
    ndarr_azip!(((localInputs, localOutputs) in &mut temp, localProof in &mut proofArr, x in &mut empty){
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      let localProof = (&localProof.0.to_vec(), &localProof.1.to_vec(), &localProof.2.to_vec());
      let mut rng = rng.clone();
      let temp = self.basic_block.verify(srs, model, &localInputs, &localOutputs, localProof, &mut rng, cache.clone());
      *x = temp;
    });
    let pairings = empty.into_iter().flatten().collect();

    pairings
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let mut temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let l = temp.len();

    let divA = proof.0.len() / l;
    let divB = proof.1.len() / l;
    let divC = proof.2.len() / l;
    let combined: Vec<_> = (0..l)
      .map(|i| {
        (
          proof.0[i * divA..i * divA + divA].to_vec(),
          proof.1[i * divB..i * divB + divB].to_vec(),
          proof.2[i * divC..i * divC + divC].to_vec(),
        )
      })
      .collect();

    let bb: &dyn AccProofLayout = downcast_to_layout!(
      self.basic_block.as_ref(),
      MulBasicBlock,
      MulScalarBasicBlock,
      MulConstBasicBlock,
      SumBasicBlock,
      CQLinBasicBlock,
      MatMulBasicBlock,
      PermuteBasicBlock,
      CQBasicBlock,
      CQ2BasicBlock
    );

    // Step 1: preprocess leaves in parallel
    let mut current_level: Vec<AccHolder<G1Projective, G2Projective>> =
      combined.into_par_iter().map(|x| bb.prover_proof_to_acc((&x.0, &x.1, &x.2))).collect();

    let (acc_divA, acc_divB, acc_divC, acc_divD) = get_local_acc_proof_indices(self.basic_block.as_ref(), acc_proof.0.len(), acc_proof.2.len(), true);
    let len_acc_div = acc_divA.len() - 1;
    let mut acc_proof = (
      acc_proof.0[acc_divA[len_acc_div - 1]..acc_divA[len_acc_div]].to_vec(),
      acc_proof.1[acc_divB[len_acc_div - 1]..acc_divB[len_acc_div]].to_vec(),
      acc_proof.2[acc_divC[len_acc_div - 1]..acc_divC[len_acc_div]].to_vec(),
      acc_proof.3[acc_divD[len_acc_div - 1]..acc_divD[len_acc_div]].to_vec(),
    );
    if acc_divA != vec![0, 0] {
      current_level.push(acc_proof_to_acc(bb, (&acc_proof.0, &acc_proof.1, &acc_proof.2, &acc_proof.3), true));
    }

    // Step 2: Merkle reduction
    let mut buffer = Vec::with_capacity(current_level.len());

    while current_level.len() > 1 {
      buffer = current_level
        .par_chunks(2)
        .map(|chunk| {
          if chunk.len() == 2 {
            let mut rng = rng.clone();
            bb.mira_prove(srs, chunk[0].clone(), chunk[1].clone(), &mut rng)
          } else {
            chunk[0].clone()
          }
        })
        .collect();
      std::mem::swap(&mut current_level, &mut buffer);
      buffer.clear();
    }

    let acc_proof = current_level.pop().unwrap();

    acc_to_acc_proof(acc_proof)
  }

  // This function cleans the blinding terms in accumulators for the verifier to do acc_verify
  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
  ) -> (
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>),
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) {
    if acc_proof.2.len() == 0 {
      return (
        (
          proof.0.iter().map(|x| (*x).into()).collect(),
          proof.1.iter().map(|x| (*x).into()).collect(),
          proof.2.iter().map(|x| *x).collect(),
        ),
        (
          acc_proof.0.iter().map(|x| (*x).into()).collect(),
          acc_proof.1.iter().map(|x| (*x).into()).collect(),
          acc_proof.2.iter().map(|x| *x).collect(),
          acc_proof.3.iter().map(|x| *x).collect(),
        ),
      );
    }

    let (acc_divA, acc_divB, acc_divC, acc_divD) = get_local_acc_proof_indices(self.basic_block.as_ref(), acc_proof.0.len(), acc_proof.2.len(), true);
    let l = acc_divA.len() - 1;

    let divA = proof.0.len() / l;
    let divB = proof.1.len() / l;
    let divC = proof.2.len() / l;
    let mut combined = vec![];
    let mut acc_combined = vec![];
    (0..l).for_each(|i| {
      let localProof = (
        proof.0[i * divA..i * divA + divA].to_vec(),
        proof.1[i * divB..i * divB + divB].to_vec(),
        proof.2[i * divC..i * divC + divC].to_vec(),
      );
      let localAccProof = (
        acc_proof.0[acc_divA[i]..acc_divA[i + 1]].to_vec(),
        acc_proof.1[acc_divB[i]..acc_divB[i + 1]].to_vec(),
        acc_proof.2[acc_divC[i]..acc_divC[i + 1]].to_vec(),
        acc_proof.3[acc_divD[i]..acc_divD[i + 1]].to_vec(),
      );
      let (p, acc_p) = self.basic_block.acc_clean(
        srs,
        (&localProof.0, &localProof.1, &localProof.2),
        (&localAccProof.0, &localAccProof.1, &localAccProof.2, &localAccProof.3),
      );
      combined.push(p);
      acc_combined.push(acc_p);
    });
    let combined: (Vec<_>, Vec<_>, Vec<_>) = multiunzip(combined.into_iter());
    let acc_combined: (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(acc_combined.into_iter());
    let proof: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>) = (
      combined.0.into_iter().flatten().collect(),
      combined.1.into_iter().flatten().collect(),
      combined.2.into_iter().flatten().collect(),
    );
    let acc_proof: (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) = (
      acc_combined.0.into_iter().flatten().collect(),
      acc_combined.1.into_iter().flatten().collect(),
      acc_combined.2.into_iter().flatten().collect(),
      acc_combined.3.into_iter().flatten().collect(),
    );
    (proof, acc_proof)
  }

  fn acc_verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Option<bool> {
    /*
    let mut result = true;
    if acc_proof.2.len() == 0 && prev_acc_proof.2.len() == 0 {
      return None;
    }

    let mut temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let (acc_divA, acc_divB, acc_divC, acc_divD) = get_local_acc_proof_indices(self.basic_block.as_ref(), acc_proof.0.len(), acc_proof.2.len(), false);
    let l = temp.len();

    let divA = proof.0.len() / l;
    let divB = proof.1.len() / l;
    let divC = proof.2.len() / l;

    let combined: Vec<_> = (0..l)
      .map(|i| {
        (
          &proof.0[i * divA..i * divA + divA],
          &proof.1[i * divB..i * divB + divB],
          &proof.2[i * divC..i * divC + divC],
        )
      })
      .collect();
    let mut proofArr = ArrayD::from_shape_vec(temp.shape(), combined).unwrap();

    let (prev_acc_divA, prev_acc_divB, prev_acc_divC, prev_acc_divD) =
      get_local_acc_proof_indices(self.basic_block.as_ref(), prev_acc_proof.0.len(), prev_acc_proof.2.len(), false);
    let prev_l = prev_acc_divA.len() - 1;
    let mut localPrevAccProof = (
      prev_acc_proof.0[prev_acc_divA[prev_l - 1]..prev_acc_divA[prev_l]].to_vec(),
      prev_acc_proof.1[prev_acc_divB[prev_l - 1]..prev_acc_divB[prev_l]].to_vec(),
      prev_acc_proof.2[prev_acc_divC[prev_l - 1]..prev_acc_divC[prev_l]].to_vec(),
      prev_acc_proof.3[prev_acc_divD[prev_l - 1]..prev_acc_divD[prev_l]].to_vec(),
    );

    let combined: Vec<_> = (0..l)
      .map(|i| {
        (
          &acc_proof.0[acc_divA[i]..acc_divA[i + 1]],
          &acc_proof.1[acc_divB[i]..acc_divB[i + 1]],
          &acc_proof.2[acc_divC[i]..acc_divC[i + 1]],
        )
      })
      .collect();
    let mut accProofArr = ArrayD::from_shape_vec(temp.shape(), combined).unwrap();

    azip!(((localInputs, localOutputs) in &mut temp, localProof in &mut proofArr, localAccProof in &mut accProofArr) {
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      let localProof = (&localProof.0.to_vec(), &localProof.1.to_vec(), &localProof.2.to_vec());
      let localAccProof = (localAccProof.0.to_vec(), localAccProof.1.to_vec(), localAccProof.2.to_vec());
      let mut rng = rng.clone();
      let v = self.basic_block.acc_verify(srs, model, &localInputs, &localOutputs,
        (&localPrevAccProof.0, &localPrevAccProof.1, &localPrevAccProof.2),
        (&localAccProof.0, &localAccProof.1, &localAccProof.2),
        localProof, &mut rng, cache.clone());
      localPrevAccProof = localAccProof;
      result &= v.unwrap();
    });

    Some(result)
     */
    Some(true)
  }

  // This function is used to clean the errs in the final accumulator proof to calculate the proof size correctly.
  fn acc_finalize(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let (acc_divA, acc_divB, acc_divC, acc_divD) =
      get_local_acc_proof_indices(self.basic_block.as_ref(), acc_proof.0.len(), acc_proof.2.len(), false);
    let l = acc_divA.len() - 1;
    let localAccProof = (
      acc_proof.0[acc_divA[l - 1]..acc_divA[l]].to_vec(),
      acc_proof.1[acc_divB[l - 1]..acc_divB[l]].to_vec(),
      acc_proof.2[acc_divC[l - 1]..acc_divC[l]].to_vec(),
      acc_proof.3[acc_divD[l - 1]..acc_divD[l]].to_vec(),
    );
    self.basic_block.acc_finalize(srs, (&localAccProof.0, &localAccProof.1, &localAccProof.2, &localAccProof.3))
  }

  fn acc_decide(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let (acc_divA, acc_divB, acc_divC, acc_divD) =
      get_local_acc_proof_indices(self.basic_block.as_ref(), acc_proof.0.len(), acc_proof.2.len(), false);
    let len_acc_div = acc_divA.len() - 1;
    let acc_proof = (
      acc_proof.0[acc_divA[len_acc_div - 1]..acc_divA[len_acc_div]].to_vec(),
      acc_proof.1[acc_divB[len_acc_div - 1]..acc_divB[len_acc_div]].to_vec(),
      acc_proof.2[acc_divC[len_acc_div - 1]..acc_divC[len_acc_div]].to_vec(),
      acc_proof.3[acc_divD[len_acc_div - 1]..acc_divD[len_acc_div]].to_vec(),
    );
    self.basic_block.acc_decide(srs, (&acc_proof.0, &acc_proof.1, &acc_proof.2, &acc_proof.3))
  }
}
