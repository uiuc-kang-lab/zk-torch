use super::BasicBlock;
use crate::{
  basic_block::{Data, DataEnc, PairingCheck, ProveVerifyCache, SRS},
  onnx,
  util::{self, calc_pow},
};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_serialize::CanonicalSerialize;
use ark_std::{cmp::max, One, UniformRand, Zero};
use ndarray::{arr0, arr1, azip, s, ArrayD, Axis};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

// SortBasicBlock is a basic block that sorts the input data in ascending or descending order.
// It takes two inputs: the data and the original indices.
// It returns three tensors: the sorted data, the sorted indice.
// Note 1: please always remember to perform one-to-one mapping check and order check after sorting.
// Note 2: we need len to be passed in as a parameter because the data may be padded with zeros, and we need to ignore the padded zeros when sorting.
#[derive(Debug)]
pub struct SortBasicBlock {
  pub descending: bool,
  pub len: usize,
}
impl BasicBlock for SortBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[1].ndim() == 1 && inputs[0].len() == inputs[1].len());
    let data = inputs[0].slice(s![..self.len]).to_owned().into_dyn();
    let data_tail = inputs[0].slice(s![self.len..]).iter().cloned().collect::<Vec<_>>();
    let indices = inputs[1].slice(s![..self.len]).to_owned().into_dyn();
    let indices_tail = inputs[1].slice(s![self.len..]).iter().cloned().collect::<Vec<_>>();

    // Pair the data and indices
    let mut paired: Vec<_> = data.into_iter().zip(indices.into_iter()).collect();
    // Sort by the first element of the tuple (data value)
    paired.sort_by_key(|&(data, _)| data);
    if self.descending {
      // Reverse the sorted data to get descending order
      paired.reverse();
    }

    // Separate the sorted data and indices
    let (sorted_data, sorted_indices): (Vec<_>, Vec<_>) = util::vec_iter(&paired).map(|(data, index)| (data, index)).unzip();
    // Concatenate the sorted data and indices with the tail
    let sorted_data = sorted_data.into_iter().chain(data_tail).collect::<Vec<_>>();
    let sorted_indices = sorted_indices.into_iter().chain(indices_tail).collect::<Vec<_>>();

    let (sorted_data, sorted_indices) = (arr1(&sorted_data).into_dyn(), arr1(&sorted_indices).into_dyn());

    vec![sorted_data, sorted_indices]
  }
}
