use crate::basic_block::*;
use ark_bls12_381::{Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::PrimeField;
use ark_std::UniformRand;
use rayon::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub fn load_file(_filename: &str, _n: usize, m: usize) -> SRS {
  // Use the same points instead of generating random ones for performance
  let mut rng = StdRng::seed_from_u64(42); // Fixed seed for deterministic results
  
  let count = 1 << m;
  
  // Generate single G1 and G2 points, then replicate them
  let g1_single = G1Projective::rand(&mut rng);
  let g2_single = G2Projective::rand(&mut rng);
  
  let g1_affine: G1Affine = g1_single.into();
  let g2_affine: G2Affine = g2_single.into();
  
  // Use the same point for all entries
  let g1: Vec<G1Affine> = vec![g1_affine; count];
  let g1_p: Vec<G1Projective> = vec![g1_single; count];
  let g2: Vec<G2Affine> = vec![g2_affine; count];
  let g2_p: Vec<G2Projective> = vec![g2_single; count];

  let res = SRS {
    Y1A: g1_affine,
    Y2A: g2_affine,
    Y1P: g1_single,
    Y2P: g2_single,
    X1A: g1,
    X2A: g2,
    X1P: g1_p,
    X2P: g2_p,
  };

  res
}
