#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{
  ops::{Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct CQBasicBlock {
  pub table_dict: HashMap<Fr, usize>,
}
impl BasicBlock for CQBasicBlock {
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert!(model.len() == 1);
    let model = model.first().unwrap();
    let N = model.raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let T_x_2 = util::msm::<G2Projective>(&srs.X2A, &model.poly.coeffs) + srs.Y2P * model.r;
    let mut temp = model.poly.coeffs[1..].to_vec();
    temp.resize(N * 2 - 1, Fr::zero());
    let mut temp2 = srs.X1P[..N].to_vec();
    temp2.reverse();
    let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
    util::fft_in_place(domain_N, &mut Q_i_x_1);
    let temp = Fr::from(N as u32).inverse().unwrap();
    let temp2 = domain_N.group_gen_inv().pow(&[(N - 1) as u64]);
    Q_i_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x *= temp * temp2.pow(&[i as u64]));
    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let temp = srs.X1P[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x = *x * domain_N.group_gen_inv().pow(&[i as u64]) - temp);
    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![T_x_2]);
  }
  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>) {
    assert!(inputs.len() == 1 && inputs[0].len() == 1);
    let model = model.first().unwrap();
    let input = inputs[0].first().unwrap();
    let N = model.raw.len();
    let n = input.raw.len();
    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    // gen(N, t):
    let Q_i_x_1 = &setup.0[..N];
    let L_i_x_1 = &setup.0[N..2 * N];
    let L_i_0_x_1 = &setup.0[2 * N..];
    if self.table_dict.len() == 0 {
      for i in 0..N {
        self.table_dict.insert(model.raw[i], i);
      }
    }

    // Calculate m
    let mut m_i = HashMap::new();
    for x in input.raw.iter() {
      if !self.table_dict.contains_key(x) {
        println!("{:?},{:?}", x, -*x);
      }
      m_i.entry(self.table_dict.get(x).unwrap()).and_modify(|y| *y += 1).or_insert(1);
    }
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[**i], Fr::from(*y as u32))).unzip();
    let m_x = util::msm::<G1Projective>(&temp, &temp2);

    let beta = Fr::rand(rng);

    // Calculate A
    let A_i: HashMap<usize, Fr> = m_i.iter().map(|(i, y)| (**i, Fr::from(*y as u32) * (model.raw[**i] + beta).inverse().unwrap())).collect();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
    let A_x = util::msm::<G1Projective>(&temp, &temp2);
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
    let A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
    let A_zero = srs.X1P[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>());
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
    let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);

    // Calculate B
    let B_i: Vec<Fr> = input.raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
    let B_poly = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_Q_poly = B_poly
      .mul(&(input.poly.clone() + (DensePolynomial { coeffs: vec![beta] })))
      .sub(&DensePolynomial { coeffs: vec![Fr::one()] })
      .divide_by_vanishing_poly(domain_n)
      .unwrap()
      .0;
    let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
    let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
    let B_zero_div = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..]);
    let B_DC = util::msm::<G1Projective>(&srs.X1A[N - n..], &B_poly.coeffs);

    let f_x_2 = util::msm::<G2Projective>(&srs.X2A, &input.poly.coeffs) + srs.Y2P * input.r;

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let mut C = vec![
      -(srs.X1P[N] - srs.X1P[0]) * r[2] + model.g1 * r[1] + A_x * model.r + (srs.Y1P * model.r * r[1]) + srs.X1P[0] * (r[1] * beta - r[0]),
      -srs.X1P[1] * r[4] + srs.X1P[0] * (r[1] - r[3]),
      -(srs.X1P[n] - srs.X1P[0]) * r[6] + input.g1 * r[5] + B_x * input.r + (srs.Y1P * input.r * r[5]) + srs.X1P[0] * (r[5] * beta),
      -srs.X1P[1] * r[7] + srs.X1P[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap()),
      -srs.X1P[0] * r[8] + srs.X1P[N - n] * r[5],
    ];
    proof.append(&mut C);

    return (proof, vec![setup.1[0].into(), f_x_2]);
  }
  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let input = inputs[0].first().unwrap();
    let model = model.first().unwrap();
    let N = model.len;
    let n = input.len;
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    let beta = Fr::rand(rng);

    // Check A(x) (A_i = m_i/(t_i+beta))
    let lhs = Bn254::pairing(A_x, T_x_2) + Bn254::pairing(A_x * beta - m_x, srs.X2A[0]);
    let rhs = Bn254::pairing(A_Q_x, srs.X2A[N] - srs.X2A[0]) + Bn254::pairing(C1, srs.Y2A);
    assert!(lhs == rhs);

    // Check T_x_2 is the G2 equivalent of the model
    let lhs = Bn254::pairing(model.g1, srs.X2A[0]);
    let rhs = Bn254::pairing(srs.X1A[0], T_x_2);
    assert!(lhs == rhs);

    // Check A(x) - A(0) is divisible by x
    let lhs = Bn254::pairing(A_x - A_zero, srs.X2A[0]);
    let rhs = Bn254::pairing(A_zero_div, srs.X2A[1]) + Bn254::pairing(C2, srs.Y2A);
    assert!(lhs == rhs);

    // Check B(x) (B_i = 1/(f_i+beta))
    let lhs = Bn254::pairing(B_x, f_x_2) + Bn254::pairing(B_x * beta - srs.X1A[0], srs.X2A[0]);
    let rhs = Bn254::pairing(B_Q_x, srs.X2A[n] - srs.X2A[0]) + Bn254::pairing(C3, srs.Y2A);
    assert!(lhs == rhs);

    // Check f_x_2 is the G2 equivalent of the input
    let lhs = Bn254::pairing(input.g1, srs.X2A[0]);
    let rhs = Bn254::pairing(srs.X1A[0], f_x_2);
    assert!(lhs == rhs);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0: G1Affine = (A_zero * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    let lhs = Bn254::pairing(B_x - B_0, srs.X2A[0]);
    let rhs = Bn254::pairing(B_zero_div, srs.X2A[1]) + Bn254::pairing(C4, srs.Y2A);
    assert!(lhs == rhs);

    // Degree check B
    let lhs = Bn254::pairing(B_x, srs.X2A[N - n]);
    let rhs = Bn254::pairing(B_DC, srs.X2A[0]) + Bn254::pairing(C5, srs.Y2A);
    assert!(lhs == rhs);
  }
}
