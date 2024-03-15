use crate::basic_block::*;
use ark_bn254::{Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::PrimeField;
use rayon::prelude::*;
use std::fs;

pub fn load_file(filename: &str, n: usize) -> SRS {
  let bytes = fs::read(filename).unwrap();

  let powers_length = 1 << n;
  let powers_g1_length = (powers_length << 1) - 1;

  let g1: Vec<G1Affine> = (0..powers_g1_length)
    .into_par_iter()
    .map(|i| {
      let start = 64 + i * 64;
      let x = Fq::from_be_bytes_mod_order(&bytes[start..start + 32]);
      let y = Fq::from_be_bytes_mod_order(&bytes[start + 32..start + 64]);
      G1Affine::new_unchecked(x, y)
    })
    .collect();
  let g1_p: Vec<G1Projective> = g1.par_iter().map(|x| (*x).into()).collect();

  let g2: Vec<G2Affine> = (0..powers_length)
    .into_par_iter()
    .map(|i| {
      let start = 64 + 64 * powers_g1_length + 128 * i;
      let a = Fq::from_be_bytes_mod_order(&bytes[start..start + 32]);
      let b = Fq::from_be_bytes_mod_order(&bytes[start + 32..start + 64]);
      let c = Fq::from_be_bytes_mod_order(&bytes[start + 64..start + 96]);
      let d = Fq::from_be_bytes_mod_order(&bytes[start + 96..start + 128]);
      G2Affine::new_unchecked(Fq2 { c0: b, c1: a }, Fq2 { c0: d, c1: c })
    })
    .collect();
  let g2_p: Vec<G2Projective> = g2.par_iter().map(|x| (*x).into()).collect();

  SRS {
    Y1A: g1[g2.len() - 1],
    Y2A: g2[g2.len() - 1],
    Y1P: g1_p[g2.len() - 1],
    Y2P: g2_p[g2.len() - 1],
    X1A: g1,
    X2A: g2,
    X1P: g1_p,
    X2P: g2_p,
  }
}
