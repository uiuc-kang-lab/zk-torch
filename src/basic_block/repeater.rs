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
impl BasicBlock for RepeaterBasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    self.basic_block.genModel()
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // Broadcast inputs to a shared larger dimension
    let dims: Vec<_> = inputs.iter().map(|input| input.shape().to_vec()).collect();
    let len = dims.iter().map(|x| x.len()).max().unwrap();
    let superDim: Vec<_> = (0..len - self.N)
      .map(|i| dims.iter().map(|dim| if dim.len() >= len - i { dim[i + dim.len() - len] } else { 1 }).max().unwrap())
      .collect();
    let broadcasted: Vec<_> = inputs
      .iter()
      .zip(dims)
      .map(|(input, dim)| {
        let mut newDim = superDim.clone();
        if dim.len() < self.N {
          newDim.extend_from_slice(&dim[..]);
        } else {
          newDim.extend_from_slice(&dim[dim.len() - self.N..]);
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
    let mut outputs = None;
    let mut outputDims = None;
    ArrayD::from_shape_fn(superDim.clone(), |idx| {
      let idx = idx.slice().to_vec();
      let mut slice: Vec<_> = idx.iter().map(|x| SliceInfoElem::Index(*x as isize)).collect();
      slice.resize(len, sliceAll);
      let subArrays: Vec<_> = broadcasted
        .iter()
        .map(|arr| {
          let sliceInfo: SliceInfo<_, IxDyn, IxDyn> = SliceInfo::try_from(&slice[..arr.ndim()]).unwrap();
          arr.slice(sliceInfo).to_owned()
        })
        .collect();
      let subArrays: Vec<_> = subArrays.iter().map(|y| y).collect();
      let localOutputs = self.basic_block.run(model, &subArrays);
      match outputs.as_mut() {
        None => {
          outputs = Some(localOutputs.iter().map(|x| x.as_slice().unwrap().to_vec()).collect::<Vec<_>>());
          outputDims = Some(localOutputs.iter().map(|x| x.shape().to_vec()).collect::<Vec<_>>());
        }
        Some(outputs) => localOutputs.iter().enumerate().for_each(|(i, x)| outputs[i].extend_from_slice(x.as_slice().unwrap())),
      }
    });
    let outputs = outputs
      .unwrap()
      .into_iter()
      .zip(outputDims.unwrap())
      .map(|(output, outputDim)| {
        let mut newOutputDim = superDim.clone();
        newOutputDim.extend_from_slice(&outputDim);
        ArrayD::from_shape_vec(newOutputDim, output).unwrap()
      })
      .collect();
    outputs
  }

  fn encodeOutputs(&self, srs: &SRS, model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    self.basic_block.encodeOutputs(srs, model, inputs, outputs)
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
    self.basic_block.prove(srs, setup, model, inputs, outputs, rng, cache)
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
    self.basic_block.verify(srs, model, inputs, outputs, proof, rng, cache)
  }
}
