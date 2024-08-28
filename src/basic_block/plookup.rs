#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::util::{self, calc_pow};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::{
  ops::{Add, Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct PlookupBasicBlock {
  pub n: usize,
}
impl BasicBlock for PlookupBasicBlock {
  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>) {
    let domain = GeneralEvaluationDomain::<Fr>::new(self.n + 1).unwrap();
    let mut L_i_x_1 = srs.X1P[..(self.n + 1)].to_vec();
    util::ifft_in_place(domain, &mut L_i_x_1);
    let mut L_i_x_2 = srs.X2P[..(self.n + 1)].to_vec();
    util::ifft_in_place(domain, &mut L_i_x_2);
    return (vec![L_i_x_1[0], L_i_x_1[self.n]], vec![L_i_x_2[0], L_i_x_2[self.n]]);
  }
  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let model = model.first().unwrap();
    let input = inputs[0].first().unwrap();
    assert!(input.raw.len() == self.n);
    let d = model.raw.len();
    let n = input.raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(n + 1).unwrap();

    // sort the input and table together to get s
    let f = &input.raw;
    let mut t = model.raw.clone();
    if d <= n {
      t.append(&mut vec![model.raw[d - 1]; n - d + 1]);
    }
    let mut s = [f.clone(), t.clone()].concat();
    s.sort();

    let t_poly = DensePolynomial { coeffs: domain.ifft(&mut t) };
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);
    let f_poly = DensePolynomial { coeffs: domain.ifft(&f) };
    let f_x = util::msm::<G1Projective>(&srs.X1A, &f_poly.coeffs);

    let mut h1 = vec![Fr::zero(); n + 1];
    let mut h2 = vec![Fr::zero(); n + 1];
    for i in 0..(n + 1) {
      h1[i] = s[i];
      h2[i] = s[n + i];
    }
    let h1_poly = DensePolynomial {
      coeffs: domain.ifft(&mut h1),
    };
    let h2_poly = DensePolynomial {
      coeffs: domain.ifft(&mut h2),
    };
    let h1_x = util::msm::<G1Projective>(&srs.X1A, &h1_poly.coeffs);
    let h2_x = util::msm::<G1Projective>(&srs.X1A, &h2_poly.coeffs);

    let gamma = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let mut Z = vec![Fr::zero(); n + 1];
    Z[0] = Fr::one();
    let mut Zi = Fr::one();
    for i in 1..(n + 1) {
      Zi *= (Fr::one() + beta)
        * (gamma + f[i - 1])
        * (gamma * (Fr::one() + beta) + t[i - 1] + beta * t[i])
        * ((gamma * (Fr::one() + beta) + s[i - 1] + beta * s[i]) * (gamma * (Fr::one() + beta) + s[n + i - 1] + beta * s[n + i]))
          .inverse()
          .unwrap();
      Z[i] = Zi;
    }
    Z[n] = Fr::one();
    let Z_poly = DensePolynomial { coeffs: domain.ifft(&mut Z) };
    let Z_x = util::msm::<G1Projective>(&srs.X1A, &Z_poly.coeffs);

    let mut L0_evals = vec![Fr::zero(); n + 1];
    L0_evals[0] = Fr::one();
    let L0_poly = DensePolynomial {
      coeffs: domain.ifft(&L0_evals),
    };
    let one = DensePolynomial { coeffs: vec![Fr::one()] };
    let L0Z_poly = L0_poly.mul(&Z_poly.sub(&one));
    let L0Z_Q = L0Z_poly.divide_by_vanishing_poly(domain).unwrap();
    let L0Z_Q_x = util::msm::<G1Projective>(&srs.X1A, &L0Z_Q.0.coeffs);

    let mut Ln_evals = vec![Fr::zero(); n + 1];
    Ln_evals[n] = Fr::one();
    let Ln_poly = DensePolynomial {
      coeffs: domain.ifft(&Ln_evals),
    };
    let LnZ_poly = Ln_poly.mul(&Z_poly.sub(&one));
    let LnZ_Q = LnZ_poly.divide_by_vanishing_poly(domain).unwrap();
    let LnZ_Q_x = util::msm::<G1Projective>(&srs.X1A, &LnZ_Q.0.coeffs);

    let h2g_poly = DensePolynomial {
      coeffs: h2_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let Lnh_poly = Ln_poly.mul(&h1_poly.sub(&h2g_poly));
    let Lnh_Q = Lnh_poly.divide_by_vanishing_poly(domain).unwrap();
    let Lnh_Q_x = util::msm::<G1Projective>(&srs.X1A, &Lnh_Q.0.coeffs);

    // for table argument, we need to still do a vanishing thing with their
    // difference
    let h1g_poly = DensePolynomial {
      coeffs: h1_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let h2g_poly = DensePolynomial {
      coeffs: h2_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let Zg_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let tg_poly = DensePolynomial {
      coeffs: t_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let gamma_poly = DensePolynomial { coeffs: vec![gamma] };
    let l1 = DensePolynomial {
      coeffs: vec![-domain.element(n), Fr::one()],
    };
    let l2 = f_poly.clone().add(gamma_poly).mul(Fr::one() + beta);
    let l3 = DensePolynomial {
      coeffs: vec![gamma * (Fr::one() + beta)],
    } + t_poly.clone()
      + tg_poly.mul(beta);
    let lhs = l3.mul(&l1).mul(&l2).mul(&Z_poly);
    let r1 = &DensePolynomial {
      coeffs: vec![gamma * (Fr::one() + beta)],
    } + &h1_poly
      + h1g_poly.mul(beta);
    let r2 = &DensePolynomial {
      coeffs: vec![gamma * (Fr::one() + beta)],
    } + &h2_poly
      + h2g_poly.mul(beta);
    let rhs = r1.mul(&l1).mul(&r2).mul(&Zg_poly);
    let Q = lhs.sub(&rhs).divide_by_vanishing_poly(domain).unwrap();
    let Q_x = util::msm::<G1Projective>(&srs.X1A, &Q.0.coeffs);

    // Multipoint opening argument for h1 + x_1 h_2 + x_1^2 t + x_1^3 Z over r, wrs
    let x_1 = Fr::rand(rng);
    let x_3 = Fr::rand(rng);
    let omega = domain.group_gen();
    let ox_3 = omega * x_3;
    // let x1s = calc_pow(x_1, 3);
    let x_1_2 = x_1 * x_1;
    let x_1_3 = x_1_2 * x_1;
    let q1_poly = &h1_poly + &h2_poly.mul(x_1) + t_poly.mul(x_1_2) + Z_poly.mul(x_1_3);
    // line interpolating q1(omega*r) and q1(r)
    let q1_r = q1_poly.evaluate(&x_3);
    let q1_or = q1_poly.evaluate(&ox_3);
    let slope = (q1_or - q1_r) * (ox_3 - x_3).inverse().unwrap();
    let r1_poly = DensePolynomial {
      coeffs: vec![q1_r - slope * x_3, slope],
    };
    assert!(r1_poly.evaluate(&x_3) == q1_r);
    assert!(r1_poly.evaluate(&ox_3) == q1_or);
    let q1_v = DensePolynomial {
      coeffs: vec![x_3 * x_3 * omega, -x_3 * (Fr::one() + omega), Fr::one()],
    };
    assert!(q1_v.evaluate(&x_3) == Fr::zero());
    assert!(q1_v.evaluate(&ox_3) == Fr::zero());
    let temp = q1_poly.sub(&r1_poly);
    let q1_Q: DensePolynomial<_> = &temp / &q1_v;
    let q1_Qx = util::msm::<G1Projective>(&srs.X1A, &q1_Q.coeffs);

    // Opening argument for f, L0h, Q over r
    let q2_poly = &f_poly + &Lnh_Q.0.mul(x_1) + Q.0.mul(x_1_2);
    let q2_r = q2_poly.evaluate(&x_3);
    let r2_poly = DensePolynomial { coeffs: vec![q2_r] };
    let q2_v = DensePolynomial {
      coeffs: vec![-x_3, Fr::one()],
    };
    let temp = q2_poly.sub(&r2_poly);
    let q2_Q = &temp / &q2_v;
    let q2_Qx = util::msm::<G1Projective>(&srs.X1A, &q2_Q.coeffs);

    // Evals
    let f_r = f_poly.evaluate(&x_3);
    let t_r = t_poly.evaluate(&x_3);
    let t_or = t_poly.evaluate(&ox_3);
    let h1_r = h1_poly.evaluate(&x_3);
    let h1_or = h1_poly.evaluate(&ox_3);
    let h2_r = h2_poly.evaluate(&x_3);
    let h2_or = h2_poly.evaluate(&ox_3);
    let Z_r = Z_poly.evaluate(&x_3);
    let Z_or = Z_poly.evaluate(&ox_3);
    let Lnh_Q_r = Lnh_Q.0.evaluate(&x_3);
    let Q_r = Q.0.evaluate(&x_3);
    let evals = vec![f_r, t_r, t_or, h1_r, h1_or, h2_r, h2_or, Z_r, Z_or, Lnh_Q_r, Q_r, q1_r, q1_or, q2_r];

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..11).map(|_| Fr::rand(&mut rng2)).collect();
    let mut C: Vec<G1Projective> = vec![
      srs.X1P[0] * (r[0] + x_1 * r[1] + x_1_2 * r[2] + x_1_3 * r[3])
        - (srs.X1P[0] * (x_3 * x_3 * omega) - srs.X1P[1] * (x_3 * (Fr::one() + omega)) + srs.X1P[2]) * r[5],
      srs.X1P[0] * (r[4] + x_1 * r[9] + x_1_2 * r[10]) - (srs.X1P[1] - (srs.X1P[0] * x_3)) * r[6],
      setup.0[0] * r[3] - (srs.X1P[n + 1] - srs.X1P[0]) * r[7],
      setup.0[1] * r[3] - (srs.X1P[n + 1] - srs.X1P[0]) * r[8],
    ];
    let proof: Vec<G1Projective> = vec![h1_x, h2_x, t_x, Z_x, f_x, q1_Qx, q2_Qx, L0Z_Q_x, LnZ_Q_x, Lnh_Q_x, Q_x];
    let mut proof: Vec<_> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    proof.append(&mut C);
    return (proof, vec![setup.1[0].into(), setup.1[1].into()], evals);
  }
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
  ) {
    let input = inputs[0].first().unwrap();
    let n = input.len;
    let domain = GeneralEvaluationDomain::<Fr>::new(n + 1).unwrap();
    let omega = domain.group_gen();
    let [h1_x, h2_x, t_x, Z_x, f_x, q1_Qx, q2_Qx, L0Z_Q_x, LnZ_Q_x, Lnh_Q_x, Q_x, C1, C2, C3, C4] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [L1, L0] = proof.1[..] else { panic!("Wrong proof format") };

    let [f_r, t_r, t_or, h1_r, h1_or, h2_r, h2_or, Z_r, Z_or, Lnh_Q_r, Q_r, q1_r, q1_or, q2_r] = proof.2[..] else {
      panic!("Wrong proof format")
    };

    let gamma = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let x_1 = Fr::rand(rng);
    let x_3 = Fr::rand(rng);

    // Check q2 commitment (h1 + x_1 h2 + x_1^2 t + x_1^3 Z)
    // let x_1_2 = x_1 * x_1;
    // let x_1_3 = x_1_2 * x_1;
    // let ox_3 = omega * x_3;
    let q1_x = h1_x + h2_x * x_1 + t_x * x_1 * x_1 + Z_x * x_1 * x_1 * x_1;
    let slope = (q1_or - q1_r) * (omega * x_3 - x_3).inverse().unwrap();
    let r1_x: G1Affine = (srs.X1P[0] * (q1_r - slope * x_3) + srs.X1P[1] * slope).into();
    let V_x: G2Affine = (srs.X2P[0] * (omega * x_3 * x_3) - srs.X2P[1] * (x_3 * (Fr::one() + omega)) + srs.X2P[2]).into();
    let lhs = Bn254::pairing(q1_x - r1_x, srs.X2A[0]);
    let rhs = Bn254::pairing(q1_Qx, V_x) + Bn254::pairing(C1, srs.Y2A);
    assert!(lhs == rhs);

    // Check f and L0h_Q commitment
    let q2_x = f_x + Lnh_Q_x * x_1 + Q_x * x_1 * x_1;
    let r2_x: G1Affine = (srs.X1P[0] * q2_r).into();
    let V_x: G2Affine = (srs.X2P[1] - srs.X2P[0] * x_3).into();
    let lhs = Bn254::pairing(q2_x - r2_x, srs.X2A[0]);
    let rhs = Bn254::pairing(q2_Qx, V_x) + Bn254::pairing(C2, srs.Y2A);
    assert!(lhs == rhs);

    // Check L0(x)(Z(x) - 1) = V(x)q(x)
    let lhs = Bn254::pairing(Z_x - srs.X1A[0], L1);
    let rhs = Bn254::pairing(L0Z_Q_x, srs.X2A[n + 1] - srs.X2A[0]) + Bn254::pairing(C3, srs.Y2A);
    assert!(lhs == rhs);

    // Check Ln(x)(Z(x) - 1) = V(x)q(x)
    let lhs = Bn254::pairing(Z_x - srs.X1A[0], L0);
    let rhs = Bn254::pairing(LnZ_Q_x, srs.X2A[n + 1] - srs.X2A[0]) + Bn254::pairing(C4, srs.Y2A);
    assert!(lhs == rhs);

    // Check Ln(r)(h1(r) - h2(wr)) = V(r)q(r)
    let mut Ln_evals = vec![Fr::zero(); n + 1];
    Ln_evals[n] = Fr::one();
    let Ln_poly = DensePolynomial {
      coeffs: domain.ifft(&Ln_evals),
    };
    let exp = [(n + 1) as u64];
    let V_r = x_3.pow(&exp) - Fr::one();
    let lhs = Ln_poly.evaluate(&x_3) * (h1_r - h2_or);
    let rhs = V_r * Lnh_Q_r;
    assert!(lhs == rhs);

    // Check Z eval
    let lhs = (x_3 - domain.element(n)) * Z_r * (Fr::one() + beta) * (gamma + f_r) * (gamma * (Fr::one() + beta) + t_r + beta * t_or);
    let rhs =
      (x_3 - domain.element(n)) * Z_or * (gamma * (Fr::one() + beta) + h1_r + beta * h1_or) * (gamma * (Fr::one() + beta) + h2_r + beta * h2_or);
    let lhs = lhs - rhs;
    let rhs = V_r * Q_r;
    assert!(lhs == rhs);
  }
}
