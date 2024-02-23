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
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

struct AProof {
  x: G1Affine,        // A(x)
  Q_x: G1Affine,      // A(x)T(x) + A(x)beta - m(x) = Q(x)Z(x)
  zero: G1Affine,     // A(0)
  zero_div: G1Affine, // (A(x)-A(0))/x
}
struct BProof {
  x: G1Affine,        // B(x)
  Q_x: G1Affine,      // B(x)f(x) + B(x)beta - 1 = Q(x)Z(x)
  zero_div: G1Affine, // (B(x)-B(0))/x
  DC: G1Affine,       // Degree Check
}

pub struct CQBasicBlock;
impl BasicBlock for CQBasicBlock {
  fn setup(&self, srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = model.raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let srs_p: Vec<G1Projective> = srs.0[..N].iter().map(|x| (*x).into()).collect();
    let T_x_2 = util::msm::<G2Projective>(&srs.1[..N], &model.poly.coeffs) + srs.1[srs.1.len() - 1] * model.r;
    let T_x_2 = T_x_2.into();
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
  fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &Data,
    inputs: &Vec<&Data>,
    _output: &Data,
    rng: &mut StdRng,
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
    for x in inputs[0].raw.iter() {
      m_i.entry(table_dict.get(x).unwrap()).and_modify(|y| *y += 1).or_insert(1);
    }
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[**i], Fr::from(*y as u32))).unzip();
    let m_x = util::msm::<G1Projective>(&temp, &temp2).into();

    let beta = Fr::rand(rng);

    // Calculate A
    let A_i: HashMap<usize, Fr> = m_i.iter().map(|(i, y)| (**i, Fr::from(*y as u32) * (model.raw[**i] + beta).inverse().unwrap())).collect();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
    let (temp3, temp4): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
    let (temp5, temp6): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
    let A = AProof {
      x: util::msm::<G1Projective>(&temp, &temp2).into(),
      Q_x: util::msm::<G1Projective>(&temp3, &temp4).into(),
      zero: (srs.0[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>())).into(),
      zero_div: util::msm::<G1Projective>(&temp5, &temp6).into(),
    };

    // Calculate B
    let B_i: Vec<Fr> = inputs[0].raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
    let B_poly = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_Q_poly = B_poly
      .mul(&(inputs[0].poly.clone() + (DensePolynomial { coeffs: vec![beta] })))
      .sub(&DensePolynomial { coeffs: vec![Fr::one()] })
      .divide_by_vanishing_poly(domain_n)
      .unwrap()
      .0;
    let B = BProof {
      x: util::msm::<G1Projective>(&srs.0, &B_poly.coeffs).into(),
      Q_x: util::msm::<G1Projective>(&srs.0, &B_Q_poly.coeffs).into(),
      zero_div: util::msm::<G1Projective>(&srs.0, &B_poly.coeffs[1..]).into(),
      DC: util::msm::<G1Projective>(&srs.0[N - n..N], &B_poly.coeffs).into(),
    };

    let f_x_2 = util::msm::<G2Projective>(&srs.1[0..n], &inputs[0].poly.coeffs) + srs.1[srs.1.len() - 1] * inputs[0].r;
    let f_x_2 = f_x_2.into();

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Affine> = vec![m_x, A.x, A.Q_x, A.zero, A.zero_div, B.x, B.Q_x, B.zero_div, B.DC];
    let mut proof: Vec<G1Affine> = proof.iter().enumerate().map(|(i, x)| ((*x) + srs.0[srs.1.len() - 1] * r[i]).into()).collect();
    let C = vec![
      -(srs.0[N] - srs.0[0]) * r[2] + model.g1 * r[1] + A.x * model.r + (srs.0[srs.1.len() - 1] * model.r * r[1]) + srs.0[0] * (r[1] * beta - r[0]),
      -srs.0[1] * r[4] + srs.0[0] * (r[1] - r[3]),
      -(srs.0[n] - srs.0[0]) * r[6]
        + inputs[0].g1 * r[5]
        + B.x * inputs[0].r
        + (srs.0[srs.1.len() - 1] * inputs[0].r * r[5])
        + srs.0[0] * (r[5] * beta),
      -srs.0[1] * r[7] + srs.0[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap()),
      -srs.0[0] * r[8] + srs.0[N - n] * r[5],
    ];
    let mut C: Vec<G1Affine> = C.iter().map(|x| (*x).into()).collect();
    proof.append(&mut C);

    return (proof, vec![setup.1[0], f_x_2]);
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    _output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let N = model.len;
    let n = inputs[0].len;
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let m_x = proof.0[0];
    let A = AProof {
      x: proof.0[1],
      Q_x: proof.0[2],
      zero: proof.0[3],
      zero_div: proof.0[4],
    };
    let B = BProof {
      x: proof.0[5],
      Q_x: proof.0[6],
      zero_div: proof.0[7],
      DC: proof.0[8],
    };
    let [C1, C2, C3, C4, C5] = proof.0[9..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };

    let beta = Fr::rand(rng);

    // Check A(x) (A_i = m_i/(t_i+beta))
    let lhs = Bn254::pairing(A.x, T_x_2) + Bn254::pairing(A.x * beta - m_x, srs.1[0]);
    let rhs = Bn254::pairing(A.Q_x, srs.1[N] - srs.1[0]) + Bn254::pairing(C1, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check T_x_2 is the G2 equivalent of the model
    let lhs = Bn254::pairing(model.g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], T_x_2);
    assert!(lhs == rhs);

    // Check A(x) - A(0) is divisible by x
    let lhs = Bn254::pairing(A.x - A.zero, srs.1[0]);
    let rhs = Bn254::pairing(A.zero_div, srs.1[1]) + Bn254::pairing(C2, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check B(x) (B_i = 1/(f_i+beta))
    let lhs = Bn254::pairing(B.x, f_x_2) + Bn254::pairing(B.x * beta - srs.0[0], srs.1[0]);
    let rhs = Bn254::pairing(B.Q_x, srs.1[n] - srs.1[0]) + Bn254::pairing(C3, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check f_x_2 is the G2 equivalent of the input
    let lhs = Bn254::pairing(inputs[0].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], f_x_2);
    assert!(lhs == rhs);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0: G1Affine = (A.zero * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    let lhs = Bn254::pairing(B.x - B_0, srs.1[0]);
    let rhs = Bn254::pairing(B.zero_div, srs.1[1]) + Bn254::pairing(C4, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Degree check B
    let lhs = Bn254::pairing(B.x, srs.1[N - n]);
    let rhs = Bn254::pairing(B.DC, srs.1[0]) + Bn254::pairing(C5, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
  }
}
