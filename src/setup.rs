use crate::util::{self, convert_to_data};
use ark_ff::Field;
use ark_std::Zero;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::basic_block::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ndarray::ArrayD;

pub struct CQLinSetup {
  pub weights: ArrayD<Data>,
  pub R: Vec<G1Affine>,
  pub Q: Vec<G1Affine>,
  pub S: Vec<G1Affine>,
  pub P_R: Vec<G1Affine>,
  pub L_V_i_x_n: Vec<G1Affine>,
  pub L_V_i_x: Vec<G1Affine>,
  pub L_H_i_x: Vec<G1Affine>,
  pub M_x: G2Affine,
}

pub struct CQSetup {
  pub table: ArrayD<Data>,
  pub Q_i_x_1: Vec<Vec<G1Affine>>,
  pub L_i_x_1: Vec<G1Affine>,
  pub L_i_0_x_1: Vec<G1Affine>,
  pub T_x_2: Vec<G2Affine>,
}

pub struct Setup {
  pub weights: HashMap<String, CQLinSetup>,
  pub tables: HashMap<String, CQSetup>,
}

impl Setup {
  fn setup_weights(srs: &SRS, weights: &ArrayD<Fr>) -> CQLinSetup {
    let weights = convert_to_data(srs, weights);
    let m = weights.len();
    let n = weights[0].raw.len();
    let N = srs.X2P.len() - 1;
    let m_inv = Fr::from(m as u64).inverse().unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_2m = GeneralEvaluationDomain::<Fr>::new(2 * m).unwrap();

    let mut L_H_i_x = srs.X1P[..n].to_vec();
    util::ifft_in_place(domain_n, &mut L_H_i_x);
    let mut L_V_i_x_n: Vec<_> = (0..m).into_par_iter().map(|i| srs.X1P[n * i]).collect();
    util::ifft_in_place(domain_m, &mut L_V_i_x_n);
    let mut L_V_i_x: Vec<G1Projective> = srs.X1P[..m].into_par_iter().map(|x| (*x).into()).collect();
    util::ifft_in_place::<G1Projective>(domain_m, &mut L_V_i_x);
    let L_H_i_x = L_H_i_x.iter().map(|x| (*x).into()).collect();
    let L_V_i_x_n = L_V_i_x_n.iter().map(|x| (*x).into()).collect();
    let L_V_i_x = L_V_i_x.iter().map(|x| (*x).into()).collect();

    // Calculate G1 U for R and S
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X1P[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate U for mn-degree check
    let mut temp: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X1P[N - m * n + i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U_P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U_P_R.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));

    // Calculate G2 U for M
    let mut temp: Vec<Vec<G2Projective>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs.X2P[i + n * j]).collect()).collect();
    temp.par_iter_mut().for_each(|x| util::ifft_in_place(domain_m, x));
    let mut U2: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| temp[j][i]).collect()).collect();
    U2.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut V = srs.X1P[m * n - n..m * n].to_vec();
    util::ifft_in_place(domain_n, &mut V);
    V.par_iter_mut().for_each(|x| *x *= m_inv);

    let mut srs_star: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| srs.X1P[n * i..n * i + n].to_vec()).collect();
    srs_star.par_iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    srs_star = (0..n).into_par_iter().map(|i| (0..m).map(|j| srs_star[m - 1 - j][i]).collect()).collect();
    srs_star.par_iter_mut().for_each(|x| x.append(&mut vec![G1Projective::zero(); m]));
    srs_star.par_iter_mut().for_each(|x| util::fft_in_place(domain_2m, x));

    let S: Vec<Vec<_>> = (0..m)
      .into_par_iter()
      .map(|i| (0..n).map(|j| (U[i][j] * domain_m.element(i).inverse().unwrap() - V[j]) * weights[i].raw[j]).collect())
      .collect();
    let S: Vec<_> = S.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    let S = S.iter().map(|x| (*x).into()).collect();

    let R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U[i][j] * weights[i].raw[j]).collect()).collect();
    let R: Vec<_> = R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    let R = R.iter().map(|x| (*x).into()).collect();

    let P_R: Vec<Vec<_>> = (0..m).into_par_iter().map(|i| (0..n).map(|j| U_P_R[i][j] * weights[i].raw[j]).collect()).collect();
    let P_R: Vec<_> = P_R.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    let P_R = P_R.iter().map(|x| (*x).into()).collect();

    // Calculate C for Q
    let mut C: Vec<Vec<_>> = (0..n).into_par_iter().map(|i| (0..m).map(|j| weights[j].raw[i]).collect()).collect();
    C.par_iter_mut().for_each(|x| domain_m.ifft_in_place(x));

    // Calculate Q. The C above corresponds to the C in the cqlin paper
    C.par_iter_mut().for_each(|x| x.append(&mut vec![Fr::zero(); m]));
    C.par_iter_mut().for_each(|x| domain_2m.fft_in_place(x));
    let temp: Vec<Vec<_>> = (0..2 * m).into_par_iter().map(|i| (0..n).map(|j| srs_star[j][i] * C[j][i]).collect()).collect();
    let mut temp: Vec<_> = temp.par_iter().map(|x| x.iter().sum::<G1Projective>()).collect();
    util::ifft_in_place(domain_2m, &mut temp);
    let mut temp = temp[m..].to_vec();
    util::fft_in_place(domain_m, &mut temp);
    let Q: Vec<_> = (0..m).into_par_iter().map(|i| temp[i] * domain_m.element(i) * m_inv).collect();
    let Q = Q.iter().map(|x| (*x).into()).collect();

    let M_x = (0..m).into_par_iter().map(|i| (0..n).map(|j| U2[i][j] * weights[i].raw[j]).sum::<G2Projective>()).sum::<G2Projective>().into(); //TODO: Change to msm

    CQLinSetup {
      weights,
      R,
      Q,
      S,
      P_R,
      L_V_i_x_n,
      L_V_i_x,
      L_H_i_x,
      M_x,
    }
  }

  fn setup_table(srs: &SRS, table: &ArrayD<Fr>) -> CQSetup {
    let data = convert_to_data(srs, table);
    let table_len = data.len();
    let table = data.view().into_shape(table_len).unwrap();
    assert!(table.ndim() == 1 && table.len() <= 2);

    let N = table[0].raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let mut Q_i_x_1s: Vec<Vec<_>> = vec![];
    let mut T_x_2 = vec![];
    for i in 0..table.len() {
      T_x_2.push((util::msm::<G2Projective>(&srs.X2A, &table[i].poly.coeffs) + srs.Y2P * table[i].r).into());
      let mut temp = table[i].poly.coeffs[1..].to_vec();
      temp.resize(N * 2 - 1, Fr::zero());
      let mut temp2 = srs.X1P[..N].to_vec();
      temp2.reverse();
      let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
      util::fft_in_place(domain_N, &mut Q_i_x_1);
      let temp = Fr::from(N as u32).inverse().unwrap();
      let temp2 = domain_N.group_gen_inv().pow(&[(N - 1) as u64]);
      Q_i_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x *= temp * temp2.pow(&[i as u64]));
      Q_i_x_1s.push(Q_i_x_1.iter().map(|x| (*x).into()).collect());
    }
    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let temp = srs.X1P[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().enumerate().for_each(|(i, x)| *x = *x * domain_N.group_gen_inv().pow(&[i as u64]) - temp);
    let L_i_x_1 = L_i_x_1.iter().map(|x| (*x).into()).collect();
    let L_i_0_x_1 = L_i_0_x_1.iter().map(|x| (*x).into()).collect();

    CQSetup {
      table: data,
      Q_i_x_1: Q_i_x_1s,
      L_i_x_1,
      L_i_0_x_1,
      T_x_2,
    }
  }

  pub fn new(srs: &SRS, weights: &HashMap<String, ArrayD<Fr>>, tables: &HashMap<String, ArrayD<Fr>>) -> Self {
    let weight_setups: HashMap<_, _> = weights.iter().map(|(k, v)| (k.clone(), Self::setup_weights(&srs, v))).collect();

    let table_setups: HashMap<_, _> = tables.iter().map(|(k, v)| (k.clone(), Self::setup_table(&srs, v))).collect();

    Setup {
      weights: weight_setups,
      tables: table_setups,
    }
  }
}
