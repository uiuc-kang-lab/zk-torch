#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{
  BasicBlock, BatchCounters, BatchProveState, BatchProveStateValues, BatchVerifyState, BatchVerifyStateValues, CacheValues, Data, DataEnc,
  PairingCheck, ProveVerifyCache, SRS,
};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{
  pairing::{Pairing, PairingOutput},
  CurveGroup,
};
use ark_ff::Field;
use ark_poly::{
  evaluations::univariate::Evaluations, univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
};
use ark_std::{
  ops::{Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::{Array1, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::{
  cell::RefCell,
  cmp::max,
  collections::HashMap,
  sync::{Arc, Mutex},
};

#[derive(Debug, Clone)]
pub struct CQBasicBlock {
  pub setup: util::CQArrayType,
}

impl BasicBlock for CQBasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    util::gen_cq_array(self.setup.clone())
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    if model.len() == 0 {
      return Ok(vec![]);
    }
    assert!(inputs.len() == 1);
    for x in inputs[0].view().as_slice().unwrap() {
      let x_int = util::fr_to_int(*x);
      if !util::check_cq_array(self.setup.clone(), x_int) {
        return Err(util::CQOutOfRangeError { input: x_int });
      }
    }
    Ok(vec![])
  }

  #[cfg(not(feature = "mock_prove"))]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    assert!(model.len() == 1);
    let model = &model.first().unwrap();
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
    let scalars: Vec<_> = (0..N).into_par_iter().map(|i| temp * temp2.pow(&[i as u64])).collect();
    util::ssm_g1_in_place(&mut Q_i_x_1, &scalars);
    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let scalars = (0..N).into_par_iter().map(|i| domain_N.group_gen_inv().pow(&[i as u64])).collect();
    util::ssm_g1_in_place(&mut L_i_0_x_1, &scalars);

    let temp = srs.X1P[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().for_each(|x| *x -= temp);

    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![T_x_2], Vec::new());
  }

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    assert!(model.len() == 1);
    let model = &model.first().unwrap();
    let N = model.raw.len();
    let L_i_x_1 = srs.X1P[..N].to_vec();
    let L_i_0_x_1 = L_i_x_1.clone();
    let Q_i_x_1 = L_i_x_1.clone();

    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, vec![srs.X2P[0]], Vec::new());
  }

  fn batch_prove_first(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    batch_prove_state: &mut BatchProveState,
    batch_counters: &mut BatchCounters,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) {
    let model = model.first().unwrap();
    let input = inputs[0].first().unwrap();
    let n = input.raw.len();
    let N = model.raw.len();
    assert!(n <= N);

    let tag = format!("{:?}_{}", self, input.raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut cache = cache.lock().unwrap();
    let CacheValues::CQTableDict(table_dict) =
      cache.entry(format!("cq_table_dict_{:p}", self)).or_insert_with(|| CacheValues::CQTableDict(HashMap::new()))
    else {
      panic!("Cache type error")
    };
    if table_dict.len() == 0 {
      for i in 0..N {
        table_dict.insert(model.raw[i], i);
      }
    }
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, N, m_i, g2s, polys, _, _, proof_2, _)) = value {
          for x in input.raw.iter() {
            if !table_dict.contains_key(x) {
              println!("{:?},{:?}", x, -*x);
            }
            m_i.borrow_mut().entry(table_dict.get(x).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
          }
          let mut p_ref = proof_2.borrow_mut();
          if p_ref.len() < 2 {
            p_ref.push(input.r);
            let af: G1Affine = (input.g1 + srs.Y1P * input.r).into();
            println!("g1 {:?}", af);
            println!("{:?}", input.r);
          }
          drop(p_ref);
          polys.borrow_mut().push(input.poly.clone());
          println!("polys {:?}", polys);
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(
              *n,
              *N,
              RefCell::clone(m_i),
              g2s.clone(),
              RefCell::clone(polys),
              vec![],
              vec![],
              RefCell::clone(proof_2),
              vec![],
            ),
          );
        } else {
        }
      }
      _ => {
        let mut m_i = HashMap::new();
        for x in input.raw.iter() {
          if !table_dict.contains_key(x) {
            println!("{:?},{:?}", x, -*x);
          }
          m_i.entry(table_dict.get(x).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
        }
        println!("inp g1 {:?}", input.g1);
        let af: G1Affine = (input.g1 + srs.Y1P * input.r).into();
        println!("g1 {:?}", af);
        println!("{:?}", input.r);
        state_mut_ref.insert(
          key,
          (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(
              n,
              N,
              RefCell::new(m_i),
              vec![setup.1[0].into()],
              RefCell::new(vec![DensePolynomial::from_coefficients_vec(vec![Fr::zero()]), input.poly.clone()]),
              vec![],
              vec![],
              RefCell::new(vec![input.r]),
              vec![],
            ),
          ),
        );
      }
    };
  }

  fn batch_prove_second(
    &self,
    _srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    batch_prove_state: &mut BatchProveState,
    batch_counters: &mut BatchCounters,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) {
    let model = model.first().unwrap();
    let input = inputs[0].first().unwrap();
    let n = input.raw.len();
    let N = model.raw.len();
    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    let tag = format!("{:?}_{}", self, input.raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, N, m_i, g2s, polys, rngs, _, proof_2, _)) = value {
          let beta = if rngs.len() == 0 { Fr::rand(rng) } else { rngs[0] };
          let B_i: Vec<Fr> = input.raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
          let agg_B = &polys.borrow()[0].clone();
          polys.borrow_mut()[0] = agg_B + &Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(
              *n,
              *N,
              RefCell::clone(m_i),
              g2s.clone(),
              RefCell::clone(polys),
              vec![beta],
              vec![],
              RefCell::clone(proof_2),
              vec![],
            ),
          )
        } else {
        }
      }
      _ => {}
    };
  }

  fn batch_prove_third(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    batch_prove_state: &mut BatchProveState,
    batch_counters: &mut BatchCounters,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) {
    let model = model.first().unwrap();
    let input = inputs[0].first().unwrap();
    let n = input.raw.len();
    let N = model.raw.len();

    // gen(N, t):
    let Q_i_x_1 = &setup.0[..N];
    let L_i_x_1 = &setup.0[N..2 * N];
    let L_i_0_x_1 = &setup.0[2 * N..];

    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let tag = format!("{:?}_{}", self, input.raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, N, m_i, g2s, polys, rngs, proof_0, proof_2, op_polys)) = value {
          println!("{} {}", N, n);
          let (proof, proof_2, f_prod, B_Q_poly, diff) = if proof_0.len() == 0 {
            let mut poly_ref = polys.borrow_mut();
            let B_poly = poly_ref[0].clone();
            let beta = rngs[0];
            let fs: Vec<_> = poly_ref[1..].iter().map(|x| x + &DensePolynomial::from_coefficients_vec(vec![beta])).collect();
            let f_prod = util::mul_polys(&fs);
            let diffs: Vec<_> = fs.iter().map(|x| &f_prod / x).collect();
            let diff = diffs.iter().fold(DensePolynomial::zero(), |acc, x| acc + x.clone());

            let mut rng2 = StdRng::from_entropy();
            let r: Vec<_> = (0..11).map(|_| Fr::rand(&mut rng2)).collect();

            let mut v_N = vec![Fr::zero(); *N + 1];
            v_N[*N] = Fr::one();
            v_N[0] = -Fr::one();
            let z_poly = DensePolynomial::from_coefficients_vec(v_N).divide_by_vanishing_poly(domain_n).unwrap().0;
            let B_zero_div = if B_poly.is_zero() {
              G1Projective::zero()
            } else {
              let prod = &B_poly.mul(&z_poly);
              util::msm::<G1Projective>(&srs.X1A, &prod.coeffs[1..])
            };

            let B_blind = DensePolynomial::from_coefficients_vec(vec![r[0]]).mul_by_vanishing_poly(domain_n);
            let B_poly = poly_ref[0].clone() + B_blind;
            let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
            let B_Q_poly = B_poly.mul(&f_prod).sub(&diff).divide_by_vanishing_poly(domain_n).unwrap().0;
            let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
            poly_ref[0] = B_poly;

            let m_ref = m_i.borrow();
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_ref.iter().map(|(i, y)| (L_i_x_1[*i], Fr::from(*y as u32))).unzip();
            let mut m_x = util::msm::<G1Projective>(&temp, &temp2);
            m_x = m_x + (srs.X1P[*N] - srs.X1P[0]) * r[1];

            let S_x = srs.X1P[1] * r[2] + (srs.X1P[*N] - srs.X1P[0]) * r[3];

            // Calculate A
            // element to m/(t + beta)
            let A_i: HashMap<usize, Fr> = m_ref.iter().map(|(i, y)| (*i, Fr::from(*y as u32) * (model.raw[*i] + beta).inverse().unwrap())).collect();
            // lagrange basis of value
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
            // msm basis with A(x)
            let mut A_x = util::msm::<G1Projective>(&temp, &temp2);
            A_x = A_x + (srs.X1P[*N] - srs.X1P[0]) * r[4];
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
            let mut A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
            A_Q_x = A_Q_x + (model.g1 + srs.X1P[0] * beta) * r[4] - srs.X1P[0] * r[1];
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
            let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);
            let Q_C_x = srs.X1P[0] * r[4] - (srs.X1P[0] * r[0]) * (Fr::from(*n as u32) * Fr::from(*N as u32).inverse().unwrap());

            let C1 = A_x * model.r;

            let commits = vec![m_x, A_x, A_Q_x, A_zero_div, B_x, B_Q_x, B_zero_div, S_x, Q_C_x, C1];
            if proof_2.borrow().len() < 2 {
              proof_2.borrow_mut().push(Fr::zero());
            }
            proof_2.borrow_mut().append(&mut vec![Fr::zero(), r[0], r[2], r[3]]);
            (commits, proof_2, f_prod, B_Q_poly, diff)
          } else {
            (
              proof_0.clone(),
              &mut RefCell::clone(proof_2),
              op_polys[1].clone(),
              op_polys[2].clone(),
              op_polys[3].clone(),
            )
          };

          let beta = rngs[0];
          let zeta = if rngs.len() <= 1 { Fr::rand(rng) } else { rngs[1] };
          let mu = if rngs.len() <= 1 { Fr::rand(rng) } else { rngs[2] };

          let mu_pow = if rngs.len() == 1 { mu } else { rngs[3] * mu };
          let f_poly = &input.poly + &DensePolynomial::from_coefficients_vec(vec![beta]);
          let q1_poly: DensePolynomial<Fr> = if op_polys.len() == 0 {
            f_poly.mul(mu_pow)
          } else {
            &op_polys[0] + &f_poly.mul(mu_pow)
          };
          if op_polys.len() == 0 {
            op_polys.append(&mut vec![q1_poly, f_prod, B_Q_poly, diff]);
          } else {
            op_polys[0] = q1_poly;
          }

          let f_z = f_poly.evaluate(&zeta);
          let agg_r = proof_2.borrow()[2];
          proof_2.borrow_mut()[2] = agg_r + input.r * mu_pow;
          proof_2.borrow_mut().push(f_z);

          let new_rngs = vec![rngs[0], zeta, mu, mu_pow];
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(
              *n,
              *N,
              RefCell::clone(m_i),
              vec![g2s[0]],
              RefCell::clone(polys),
              new_rngs,
              proof,
              RefCell::clone(proof_2),
              op_polys.clone(),
            ),
          )
        } else {
        }
      }
      _ => {}
    }
  }

  fn batch_prove(&self, srs: &SRS, batch_prove_values: &BatchProveStateValues, _rng: &mut StdRng) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    if let BatchProveStateValues::CQ(n, N, _, g2s, polys, rngs, proof_0, proof_2, op_polys) = batch_prove_values {
      let domain_n = GeneralEvaluationDomain::<Fr>::new(*n).unwrap();
      let beta = rngs[0];
      let zeta = rngs[1];
      let mu = rngs[2];
      println!("beta {:?}", beta);
      println!("zeta {:?}", zeta);
      println!("mu {:?}", mu);

      let [m_x, A_x, A_Q_x, A_zero_div, B_x, B_Q_x, B_zero_div, S_x, Q_C_x, C1] = proof_0[..] else {
        panic!("Wrong proof format")
      };
      let p_ref = proof_2.borrow();
      let R_C_x = A_zero_div - B_zero_div * (Fr::from(*n as u32) * Fr::from(*N as u32).inverse().unwrap()) + srs.X1P[0] * (mu * p_ref[3]);
      let Q_x = A_Q_x + Q_C_x * mu + srs.X1P[0] * mu * mu * p_ref[4];
      let poly_ref = polys.borrow();
      let B_poly = &poly_ref[0];
      let fs: Vec<_> = poly_ref[1..].iter().map(|x| x + &DensePolynomial::from_coefficients_vec(vec![beta])).collect();
      let f_zs = &p_ref[5..];
      let q1_poly = &op_polys[0];
      let q1_z = q1_poly.evaluate(&zeta);
      let q1_z_poly = DensePolynomial { coeffs: vec![q1_z] };
      let q1_V = DensePolynomial {
        coeffs: vec![-zeta, Fr::one()],
      };
      let W_poly = &q1_poly.sub(&q1_z_poly) / &q1_V;
      let W_x = util::msm::<G1Projective>(&srs.X1A, &W_poly.coeffs);

      let fz_prod: Fr = f_zs[1..].iter().product();
      let diffs: Vec<_> = f_zs[1..].iter().map(|x| &fz_prod / x).collect();
      let diff_poly = if fs.len() == 1 {
        DensePolynomial::from_coefficients_vec(vec![Fr::one()])
      } else if fs.len() == 2 {
        &fs[0] + &fs[1]
      } else {
        &(fs[1].mul(&DensePolynomial::from_coefficients_vec(vec![diffs[0]])))
          + &(fs[0].mul(&DensePolynomial::from_coefficients_vec(vec![diffs
            .iter()
            .fold(Fr::zero(), |acc, x| acc + x.clone())])))
      };
      println!("{:?} {:?}", p_ref[0], p_ref[1]);
      let diff_r = if fs.len() == 1 {
        Fr::zero()
      } else if fs.len() == 2 {
        p_ref[0] + p_ref[1]
      } else {
        p_ref[1] * diffs[0] + p_ref[0] * diffs.iter().fold(Fr::zero(), |acc, x| acc + x.clone())
      };

      let f_prod = &op_polys[1];
      let B_Q_poly = &op_polys[2];
      let v_poly = domain_n.vanishing_polynomial();
      let v_z = v_poly.evaluate(&zeta);
      let f_prod_z = f_prod.evaluate(&zeta);

      let C2 = -srs.X1A[0] * diff_r;
      let D_poly = &(&B_poly.mul(&DensePolynomial::from_coefficients_vec(vec![f_prod_z])) - &diff_poly)
        - &B_Q_poly.mul(&DensePolynomial::from_coefficients_vec(vec![v_z]));
      let P_poly = D_poly;
      let P_poly = &P_poly / &DensePolynomial::from_coefficients_vec(vec![-zeta, Fr::one()]);
      let P_x = util::msm::<G1Projective>(&srs.X1A, &P_poly.coeffs);
      let r_sum = p_ref[2];
      let C3 = srs.X1P[0] * r_sum;

      let proof = vec![m_x, A_x, B_x, B_Q_x, S_x, P_x, R_C_x, Q_x, W_x, C1, C2, C3];
      let proof_1 = g2s.clone();
      (proof, proof_1, f_zs.to_vec())
    } else {
      (vec![], vec![], vec![])
    }
  }

  fn batch_verify_first(
    &self,
    batch_verify_state: &mut BatchVerifyState,
    batch_counters: &mut BatchCounters,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) {
    let input = inputs[0].first().unwrap();
    let model = model.first().unwrap();
    let tag = format!("{:?}_{}", self, input.len);
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut state_mut_ref = batch_verify_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        let (_, BatchVerifyStateValues::CQ(n, N, _, g1s, beta)) = value;
        let mut new_g1s = g1s.clone();
        new_g1s.push(input.g1);
        *value = (Box::new(self.clone()), BatchVerifyStateValues::CQ(*n, *N, model.g1, new_g1s, *beta))
      }
      _ => {
        let N = model.len;
        let beta = Fr::rand(rng);
        state_mut_ref.insert(
          key,
          (
            Box::new(self.clone()),
            BatchVerifyStateValues::CQ(input.len, N, model.g1, vec![input.g1], beta),
          ),
        );
      }
    };
  }

  fn batch_verify(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    batch_verify_values: &BatchVerifyStateValues,
    rng: &mut StdRng,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let [m_x, A_x, B_x, B_Q_x, S_x, P_x, R_C_x, Q_x, W_x, C1, C2, C3] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2] = proof.1[..] else { panic!("Wrong proof format") };
    let f_zs = proof.2;
    let zeta = Fr::rand(rng);
    let mu = Fr::rand(rng);

    let BatchVerifyStateValues::CQ(n, N, model_g1, input_xs, beta) = batch_verify_values;
    println!("beta {:?}", beta);
    println!("zeta {:?}", zeta);
    println!("mu {:?}", mu);

    let domain_n = GeneralEvaluationDomain::<Fr>::new(*n).unwrap();
    let m = f_zs.len();
    let mus = util::calc_pow(mu, m);
    let fz_sum: Fr = f_zs.iter().enumerate().map(|(i, x)| *x * mus[i]).sum();
    let f_xs: Vec<_> = input_xs.iter().map(|x| (*x + srs.X1A[0] * beta)).collect();
    let f_sum: G1Projective = f_xs.iter().enumerate().map(|(i, x)| *x * mus[i]).sum();
    let fz_prod: Fr = f_zs.iter().product();
    let fz_prod_1: Fr = f_zs[1..].iter().product();
    let diffs: Vec<_> = f_zs[1..].iter().map(|x| &fz_prod_1 / x).collect();
    println!("len {} {}", f_zs.len(), f_xs.len());
    let diff_x = if f_xs.len() == 1 {
      srs.X1A[0] * Fr::one()
    } else if f_xs.len() == 2 {
      f_xs[0] + f_xs[1]
    } else {
      f_xs[1] * diffs[0] + f_xs[0] * diffs.iter().fold(Fr::zero(), |acc, x| acc + x.clone())
    };
    let v_poly = domain_n.vanishing_polynomial();
    let v_z = v_poly.evaluate(&zeta);
    let D_x: G1Affine = (B_x * fz_prod - diff_x - B_Q_x * v_z).into();

    // Check A(x) (A_i = m_i/(t_i+beta))
    let mut v_N = vec![Fr::zero(); N + 1];
    v_N[*N] = Fr::one();
    v_N[0] = -Fr::one();
    let z_poly = DensePolynomial::from_coefficients_vec(v_N).divide_by_vanishing_poly(domain_n).unwrap().0;
    let z_x = util::msm::<G2Projective>(&srs.X2A, &z_poly.coeffs);
    checks.push(vec![
      (A_x, T_x_2),
      ((A_x * (*beta + mu) - m_x + S_x * mu * mu).into(), srs.X2A[0]),
      (
        (-B_x * mu * Fr::from(*n as u32) * Fr::from(*N as u32).inverse().unwrap()).into(),
        z_x.into(),
      ),
      ((-Q_x, (srs.X2A[*N] - srs.X2A[0]).into())),
      ((-R_C_x * mu).into(), srs.X2A[1]),
      (-C1, srs.Y2A),
    ]);

    // Check T_x_2 is the G2 equivalent of the model
    checks.push(vec![(*model_g1, srs.X2A[0]), (srs.X1A[0], -T_x_2)]);

    // Check D(x) (D(x) = B(x) * F(x) - diff(x) - B_Q(x) * v(x))
    let zeta_x: G2Affine = (srs.X2A[0] * zeta).into();
    if f_xs.len() == 1 {
      checks.push(vec![((D_x, srs.X2A[0])), (-P_x, (srs.X2A[1] - zeta_x).into())]);
    } else {
      checks.push(vec![((D_x, srs.X2A[0])), (-P_x, (srs.X2A[1] - zeta_x).into()), (-C2, srs.Y2A)]);
    };

    // Check W(x) (W(x) = (F(x) - F(zeta)) / (x - zeta))
    // where F(x) is RLC'd input polynomials
    checks.push(vec![
      ((-W_x, (srs.X2P[1] - srs.X2A[0] * zeta).into())),
      ((f_sum - srs.X1A[0] * fz_sum).into(), srs.X2A[0]),
      (-C3, srs.Y2A),
    ]);

    checks
  }
}
