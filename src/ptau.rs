use crate::basic_block::*;
use ark_bls12_381::{Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::PrimeField;
use ark_std::UniformRand;
use rayon::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub fn load_file(_filename: &str, _n: usize, m: usize) -> SRS {
  // Generate random SRS elements instead of loading from challenge file
  let mut rng = StdRng::seed_from_u64(42); // Fixed seed for deterministic results
  
  let count = 1 << m;
  
  // Generate random G1 points
  let g1: Vec<G1Affine> = (0..count)
    .into_par_iter()
    .map(|i| {
      let mut local_rng = StdRng::seed_from_u64(42 + i as u64);
      G1Projective::rand(&mut local_rng).into()
    })
    .collect();
  let g1_p: Vec<G1Projective> = g1.par_iter().map(|x| (*x).into()).collect();

  // Generate random G2 points
  let g2: Vec<G2Affine> = (0..count)
    .into_par_iter()
    .map(|i| {
      let mut local_rng = StdRng::seed_from_u64(1000 + i as u64);
      G2Projective::rand(&mut local_rng).into()
    })
    .collect();
  let g2_p: Vec<G2Projective> = g2.par_iter().map(|x| (*x).into()).collect();

  let res = SRS {
    Y1A: g1[g2.len() - 1],
    Y2A: g2[g2.len() - 1],
    Y1P: g1_p[g2.len() - 1],
    Y2P: g2_p[g2.len() - 1],
    X1A: g1,
    X2A: g2,
    X1P: g1_p,
    X2P: g2_p,
  };

  res
}
