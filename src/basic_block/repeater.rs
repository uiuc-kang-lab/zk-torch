use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr1, azip, s, ArrayD, Axis, Dimension, IxDyn, SliceInfo, SliceInfoElem};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct RepeaterBasicBlock {
  pub basic_block: Box<dyn BasicBlock>,
  pub N: usize,
}

fn broadcastN<T1: Clone, T2: Clone>(
  inputs: &Vec<&ArrayD<T1>>,
  outputs: Option<&Vec<&ArrayD<T2>>>,
  N: usize,
) -> ArrayD<(Vec<ArrayD<T1>>, Option<Vec<ArrayD<T2>>>)> {
  // Broadcast inputs to a shared larger dimension
  let dims: Vec<_> = inputs.iter().map(|input| input.shape().to_vec()).collect();
  let len = dims.iter().map(|x| x.len()).max().unwrap();
  let superDim: Vec<_> = (0..len - N)
    .map(|i| dims.iter().map(|dim| if dim.len() >= len - i { dim[i + dim.len() - len] } else { 1 }).max().unwrap())
    .collect();
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

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    let temp = broadcastN::<Fr, Fr>(inputs, None, self.N);
    let temp = temp.map(|(subArrays, _)| {
      let subArrays: Vec<_> = subArrays.iter().map(|y| y).collect();
      self.basic_block.run(model, &subArrays)
    });
    let temp = temp.map(|x| x.iter().map(|y| y).collect());
    let temp = temp.map(|x| x);
    combineArr(&temp)
  }

  fn encodeOutputs(&self, srs: &SRS, model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let temp = temp.map(|(localInputs, localOutputs)| {
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      self.basic_block.encodeOutputs(srs, model, &localInputs, &localOutputs)
    });
    let temp = temp.map(|x| x.iter().map(|y| y).collect());
    let temp = temp.map(|x| x);
    combineArr(&temp)
  }

  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    self.basic_block.setup(srs, model)
  }

  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let temp = broadcastN(inputs, Some(outputs), self.N - 1);
    let temp = temp.map(|(localInputs, localOutputs)| {
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      self.basic_block.prove(srs, setup, model, &localInputs, &localOutputs, rng, cache)
    });
    let mut proof = (vec![], vec![]);
    temp.for_each(|(a, b)| {
      proof.0.extend_from_slice(&a[..]);
      proof.1.extend_from_slice(&b[..]);
    });
    proof
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
    cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let temp = broadcastN(inputs, Some(outputs), self.N - 1);

    let l = temp.len();
    let divA = proof.0.len() / l;
    let divB = proof.1.len() / l;
    let combined: Vec<_> = (0..l).map(|i| (&proof.0[i * divA..i * divA + divA], &proof.1[i * divB..i * divB + divB])).collect();
    let proofArr = ArrayD::from_shape_vec(temp.shape(), combined).unwrap();

    let mut pairings = vec![];
    azip!(((localInputs, localOutputs) in &temp, localProof in &proofArr){
      let localInputs: Vec<_> = localInputs.iter().map(|y| y).collect();
      let localOutputs: Vec<_> = localOutputs.as_ref().unwrap().iter().map(|y| y).collect();
      let localProof = (&localProof.0.to_vec(), &localProof.1.to_vec());
      let mut temp = self.basic_block.verify(srs, model, &localInputs, &localOutputs, localProof, rng, cache);
      pairings.append(&mut temp);
    });
    pairings
  }
}
