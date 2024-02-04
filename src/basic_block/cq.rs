#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{
  ops::{Mul, Sub},
  One, UniformRand, Zero,
};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct CQBasicBlock;
impl BasicBlock for CQBasicBlock {
  fn setup(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = model.raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let srs_p: Vec<G1Projective> = srs.0[..N].iter().map(|x| (*x).into()).collect();
    let T_x_2 = util::msm::<G2Projective>(&srs.1[..N], &model.poly.coeffs).into();
    let mut temp = model.poly.coeffs[1..].to_vec();
    temp.resize(N * 2 - 1, Fr::zero());
    let mut temp2 = srs_p.to_vec();
    temp2.reverse();
    let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
    util::fft_in_place(domain_N, &mut Q_i_x_1);
    let temp = Fr::from(N as u32).inverse().unwrap();
    let temp2 = domain_N.group_gen_inv().pow(&[(N - 1) as u64]);
    Q_i_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x *= temp * temp2.pow(&[i as u64]));
    let mut L_i_x_1 = srs_p;
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let temp = srs.0[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x = *x * domain_N.group_gen_inv().pow(&[i as u64]) - temp);
    let Q_i_x_1: Vec<G1Affine> = Q_i_x_1.par_iter().map(|x| (*x).into()).collect();
    let L_i_x_1: Vec<G1Affine> = L_i_x_1.par_iter().map(|x| (*x).into()).collect();
    let L_i_0_x_1: Vec<G1Affine> = L_i_0_x_1.par_iter().map(|x| (*x).into()).collect();
    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![T_x_2]);
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Data,
    inputs: &Vec<Data>,
    _output: &Data,
    rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = model.raw.len();
    let n = inputs[0].raw.len();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    // gen(N, t):
    let Q_i_x_1 = &setup.0[..N];
    let L_i_x_1 = &setup.0[N..2 * N];
    let L_i_0_x_1 = &setup.0[2 * N..];
    let mut table_dict = HashMap::new();
    for i in 0..N {
      table_dict.insert(model.raw[i], i);
    }

    // Calculate m
    let mut m_i = HashMap::new();
    for i in 0..n {
      m_i.entry(table_dict.get(&inputs[0].raw[i]).unwrap()).and_modify(|x| *x += 1).or_insert(1);
    }
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[**i], Fr::from(*y as u32))).unzip();
    let m_x_1 = util::msm::<G1Projective>(&temp, &temp2).into();

    let beta = Fr::rand(rng);

    // Calculate A
    let A_i: HashMap<usize, Fr> = m_i.iter().map(|(i, y)| (**i, Fr::from(*y as u32) * (model.raw[**i] + beta).inverse().unwrap())).collect();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
    let A_x_1 = util::msm::<G1Projective>(&temp, &temp2).into();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
    let Q_A_x_1 = util::msm::<G1Projective>(&temp, &temp2).into();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
    let A_0_x_1 = util::msm::<G1Projective>(&temp, &temp2).into();
    let A_0 = Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>();
    let A_0_1 = (srs.0[0] * A_0).into();

    // Calculate B
    let B_i: Vec<Fr> = (0..n).map(|i| (inputs[0].raw[i] + beta).inverse().unwrap()).collect();
    let B = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_x_1 = util::msm::<G1Projective>(&srs.0, &B.coeffs).into();
    let mut Q_B = B.mul(&(inputs[0].poly.clone() + (DensePolynomial { coeffs: vec![beta] })));
    Q_B = Q_B.sub(&DensePolynomial { coeffs: vec![Fr::one()] }).divide_by_vanishing_poly(domain_n).unwrap().0;
    let Q_B_x_1 = util::msm::<G1Projective>(&srs.0, &Q_B).into();
    let B_0_x_1 = util::msm::<G1Projective>(&srs.0, &B.coeffs[1..]).into();
    let B_DC = util::msm::<G1Projective>(&srs.0[N - n..N], &B.coeffs).into();

    let f_x_2 = util::msm::<G2Projective>(&srs.1[0..n], &inputs[0].poly.coeffs).into();

    return (
      vec![m_x_1, A_x_1, Q_A_x_1, A_0_1, A_0_x_1, B_x_1, Q_B_x_1, B_0_x_1, B_DC],
      vec![setup.1[0], f_x_2],
    );
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<DataEnc>,
    _output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut R,
  ) {
    let N = model.dims[0];
    let n = inputs[0].dims[0];
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let [m_x_1, A_x_1, Q_A_x_1, A_0_1, A_0_x_1, B_x_1, Q_B_x_1, B_0_x_1, B_DC] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    let beta = Fr::rand(rng);

    // Check A_x_1 (A_i = m_i/(t_i+beta))
    let lhs = Bn254::pairing(A_x_1, T_x_2) + Bn254::pairing(A_x_1 * beta - m_x_1, srs.1[0]);
    let rhs = Bn254::pairing(Q_A_x_1, srs.1[N] - srs.1[0]);
    assert!(lhs == rhs);

    // Check T_x_2 is the G2 equivalent of T_x_1
    let lhs = Bn254::pairing(model.g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], T_x_2);
    assert!(lhs == rhs);

    // Check A(x) - A(0) is divisible by x
    let lhs = Bn254::pairing(A_x_1 - A_0_1, srs.1[0]);
    let rhs = Bn254::pairing(A_0_x_1, srs.1[1]);
    assert!(lhs == rhs);

    // Check B_x_1 (B_i = 1/(f_i+beta))
    let lhs = Bn254::pairing(B_x_1, f_x_2) + Bn254::pairing(B_x_1 * beta - srs.0[0], srs.1[0]);
    let rhs = Bn254::pairing(Q_B_x_1, srs.1[n] - srs.1[0]);
    assert!(lhs == rhs);

    // Check f_x_2 is the G2 equivalent of f_x_1
    let lhs = Bn254::pairing(inputs[0].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], f_x_2);
    assert!(lhs == rhs);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0_1: G1Affine = (A_0_1 * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    let lhs = Bn254::pairing(B_x_1 - B_0_1, srs.1[0]);
    let rhs = Bn254::pairing(B_0_x_1, srs.1[1]);

    // Degree check B
    let lhs = Bn254::pairing(B_x_1, srs.1[N - n]);
    let rhs = Bn254::pairing(B_DC, srs.1[0]);

    assert!(lhs == rhs);
  }
}
