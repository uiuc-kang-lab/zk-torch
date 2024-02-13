#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain, Polynomial};
use ark_bn254::{Fr, G1Projective, G2Projective, G1Affine, G2Affine, Bn254};
use ark_std::{Zero, One, UniformRand};
use rayon::prelude::*;
use rand::Rng;
use ndarray::{Array, IxDyn};
use super::{BasicBlock,Data,DataEnc,Tensor};
use crate::util;

pub struct CQLinBasicBlock;
impl BasicBlock for CQLinBasicBlock{
  type Proof = (Vec<G1Affine>, Vec<G2Affine>);
  type Setup = (Vec<G1Affine>, Vec<G2Affine>);
  fn run(
    model: &Vec<Tensor<Fr>>,
    inputs: &Vec<Tensor<Fr>>,
  ) -> Vec<Tensor<Fr>> {
    let n = inputs[0].len();
    let mut r = vec![Fr::zero(); n];
    for i in 0..n {
      for j in 0..n {
        r[i] += model[0][[j * n + i]] * inputs[0][j];
      }
    }
    vec![Array::from_shape_vec(IxDyn(&[n]), r).unwrap()]
  }
  fn setup(srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = model.raw.len();
    let n: usize = (N as f64).sqrt() as usize;
    let n_inv = Fr::from(n as u64).inverse().unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2n = GeneralEvaluationDomain::<Fr>::new(2 * n).unwrap();
    let srs_p: Vec<G1Projective> = srs.0[..N].par_iter().map(|x| (*x).into()).collect();
    let mut L_i_x = srs_p[..n].to_vec();
    util::ifft_in_place(domain_n, &mut L_i_x);
    let mut L_i_x_n: Vec<_> = (0..n).into_par_iter().map(|i| srs_p[n * i]).collect();
    util::ifft_in_place(domain_n, &mut L_i_x_n);

    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..n).map(|j| srs_p[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut U: Vec<Vec<_>> = (0..n).into_par_iter().map(|j| (0..n).map(|i| temp[i][j]).collect()).collect();
    U.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut temp: Vec<Vec<G2Projective>> = (0..n).into_par_iter().map(|i| (0..n).map(|j| srs.1[i + n * j].into()).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut U2: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U2.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut V = srs_p[N - n..N].to_vec();
    util::ifft_in_place(domain_n, &mut V);
    V.par_iter_mut().for_each(|x| *x *= n_inv);

    let mut srs_star: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| srs_p[n * i..n * i + n].to_vec()).collect();
    srs_star.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    srs_star = (0..n).into_par_iter().map(|i| (0..n).map(|j| srs_star[n - 1 - j][i]).collect()).collect();
    srs_star.par_iter_mut().for_each(|x| x.append(&mut vec![G1Projective::zero(); n]));
    srs_star.par_iter_mut().for_each(|x| util::fft_in_place(domain_2n, x));

    let mut Ls = vec![vec![Fr::zero(); n]; n];
    Ls.par_iter_mut().enumerate().for_each(|(i, x)| x[i] = Fr::one());
    Ls.par_iter_mut().for_each(|x| domain_n.ifft_in_place(x));
    let S: Vec<Vec<_>> = (0..n)
      .into_par_iter()
      .map(|i| (0..n).map(|j| (U[i][j] * domain_n.element(i).inverse().unwrap() - V[j]) * model.raw[i * n + j]).collect())
      .collect();
    let S: Vec<_> = S.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    let R: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..n).map(|j| U[i][j] * model.raw[i * n + j]).collect()).collect();
    let R: Vec<_> = R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();

    let mut C: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..n).map(|j| model.raw[j * n + i]).collect()).collect();
    C.par_iter_mut().for_each(|x| domain_n.ifft_in_place(x));

    let mut temp = C;
    temp.par_iter_mut().for_each(|x| x.append(&mut vec![Fr::zero(); n]));
    temp.par_iter_mut().for_each(|x| domain_2n.fft_in_place(x));
    let temp: Vec<Vec<_>> = (0..2 * n).into_par_iter().map(|i| (0..n).map(|j| srs_star[j][i] * temp[j][i]).collect()).collect();
    let mut temp: Vec<_> = temp.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    util::ifft_in_place(domain_2n, &mut temp);
    let mut temp = temp[n..].to_vec();
    util::fft_in_place(domain_n, &mut temp);
    let Q: Vec<_> = (0..n).into_par_iter().map(|i| temp[i] * domain_n.element(i) * n_inv).collect();
    let M_x = (0..n).into_par_iter().map(|i| (0..n).map(|j| U2[i][j] * model.raw[i * n + j]).sum::<G2Projective>()).sum::<G2Projective>(); //TODO: Change to msm

    let R: Vec<G1Affine> = R.par_iter().map(|x| (*x).into()).collect();
    let mut Q: Vec<G1Affine> = Q.par_iter().map(|x| (*x).into()).collect();
    let mut S: Vec<G1Affine> = S.par_iter().map(|x| (*x).into()).collect();
    let mut L_i_x: Vec<G1Affine> = L_i_x.par_iter().map(|x| (*x).into()).collect();
    let mut L_i_x_n: Vec<G1Affine> = L_i_x_n.par_iter().map(|x| (*x).into()).collect();
    let mut setup = R;
    setup.append(&mut Q);
    setup.append(&mut S);
    setup.append(&mut L_i_x);
    setup.append(&mut L_i_x_n);
    return (setup, vec![M_x.into()]);
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: &Self::Setup,
    _model: &Data,
    inputs: &Vec<Data>,
    output: &Data,
    rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let n = inputs[0].raw.len();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let R = &setup.0[..n];
    let Q = &setup.0[n..2 * n];
    let S = &setup.0[2 * n..3 * n];
    let L_i_x = &setup.0[3 * n..4 * n];
    let L_i_x_n = &setup.0[4 * n..];

    let inputs_0_vec = inputs[0].raw.clone().into_iter().collect::<Vec<_>>();
    let R_x = util::msm::<G1Projective>(R, &inputs_0_vec).into();
    let Q_x = util::msm::<G1Projective>(Q, &inputs_0_vec).into();
    let temp: Vec<_> = (0..n).into_par_iter().map(|i|srs.0[n * i]).collect();
    let A_x = util::msm::<G1Projective>(&temp, &inputs[0].poly.coeffs).into();
    let S_x = util::msm::<G1Projective>(S, &inputs_0_vec).into();
    let P_x = util::msm::<G1Projective>(&srs.0[n * n - n..n * n], &output.poly.coeffs).into();

    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let z = inputs[0].poly.evaluate(&gamma_n);
    let h_i: Vec<_> = (0..n).into_par_iter().map(|i| (inputs[0].raw[i] - z) * (domain_n.element(i) - gamma_n).inverse().unwrap()).collect();
    let z = (srs.0[0] * z).into();
    let pi = util::msm::<G1Projective>(&L_i_x, &h_i).into();
    let pi_1 = util::msm::<G1Projective>(&L_i_x_n, &h_i).into();

    return (vec![R_x, Q_x, A_x, S_x, P_x, z, pi, pi_1], vec![setup.1[0]]);
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    proof: &Self::Proof,
    rng: &mut R,
  ) {
    let n = inputs[0].len;
    let [R_x, Q_x, A_x, S_x, P_x, z, pi, pi_1] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let lhs = Bn254::pairing(A_x, proof.1[0]);
    let rhs = Bn254::pairing(Q_x, srs.1[n * n] - srs.1[0]) + Bn254::pairing(R_x, srs.1[0]);
    assert!(lhs == rhs);

    let temp: G1Affine = (output.g1 * Fr::from(n as u64).inverse().unwrap()).into();
    let lhs = Bn254::pairing(R_x - temp, srs.1[0]);
    let rhs = Bn254::pairing(S_x, srs.1[n]);
    assert!(lhs == rhs);

    let lhs = Bn254::pairing(output.g1, srs.1[n * n - n]);
    let rhs = Bn254::pairing(P_x, srs.1[0]);
    assert!(lhs == rhs);

    let gamma = Fr::rand(rng);
    let gamma_n = gamma.pow(&[n as u64]);
    let lhs = Bn254::pairing(inputs[0].g1 - z + pi * gamma_n, srs.1[0]);
    let rhs = Bn254::pairing(pi, srs.1[1]);
    assert!(lhs == rhs);

    let lhs = Bn254::pairing(A_x - z + pi_1 * gamma_n, srs.1[0]);
    let rhs = Bn254::pairing(pi_1, srs.1[n]);
    assert!(lhs == rhs);
  }
}
