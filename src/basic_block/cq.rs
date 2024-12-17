#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{
  BasicBlock, BatchProveState, BatchProveStateValues, BatchVerifyState, BatchVerifyStateValues, CacheValues, Data, DataEnc, PairingCheck,
  ProveVerifyCache, SRS,
};
use crate::util;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
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
use std::{cmp::max, collections::HashMap};

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

    let key = format!("{:?}_{}", self, input.raw.len());
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
    let mut guard = batch_prove_state.lock().unwrap();
    match guard.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, m_i, T_x_2, polys, _, _, _)) = value {
          let mut m_i = m_i.clone();
          for x in input.raw.iter() {
            if !table_dict.contains_key(x) {
              println!("{:?},{:?}", x, -*x);
            }
            m_i.entry(table_dict.get(x).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
          }
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(*n, m_i, *T_x_2, polys.clone(), vec![], vec![], vec![]),
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
        guard.insert(
          key,
          (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(
              n,
              m_i,
              setup.1[0].into(),
              vec![
                DensePolynomial::from_coefficients_vec(vec![Fr::zero()]),
                DensePolynomial::from_coefficients_vec(vec![Fr::one()]),
                DensePolynomial::from_coefficients_vec(vec![Fr::zero()]),
                DensePolynomial::from_coefficients_vec(vec![Fr::zero()]),
              ],
              vec![],
              vec![],
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

    let key = format!("{:?}_{}", self, input.raw.len());
    let mut guard = batch_prove_state.lock().unwrap();
    match guard.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, m_i, T_x_2, polys, rngs, _, _)) = value {
          let beta = if rngs.len() == 0 { Fr::rand(rng) } else { rngs[0] };
          let B_i: Vec<Fr> = input.raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
          let f_poly = input.poly.clone() + DensePolynomial::from_coefficients_vec(vec![beta]);
          let B_poly = &polys[0] + &Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
          let f_prod = &polys[1];
          let diff = &polys[2].mul(&f_poly) + f_prod;
          let f_prod = f_poly.mul(f_prod);
          let new_polys = vec![B_poly, f_prod, diff];
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(*n, m_i.clone(), *T_x_2, new_polys, vec![beta], vec![], vec![]),
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
    let key = format!("{:?}_{}", self, input.raw.len());
    let mut guard = batch_prove_state.lock().unwrap();
    match guard.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ(n, m_i, T_x_2, polys, rngs, proof_0, proof_2)) = value {
          let (proof, mut proof_2) = if proof_0.len() == 0 {
            let B_poly = &polys[0];
            let f_prod = &polys[1];
            let diff = &polys[2];

            let B_Q_poly = B_poly.mul(f_prod).sub(diff).divide_by_vanishing_poly(domain_n).unwrap().0;
            let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
            let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
            let B_zero_div = if B_poly.is_zero() {
              G1Projective::zero()
            } else {
              util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..])
            };
            let B_DC = util::msm::<G1Projective>(&srs.X1A[N - *n..], &B_poly.coeffs);

            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[*i], Fr::from(*y as u32))).unzip();
            let m_x = util::msm::<G1Projective>(&temp, &temp2);

            // Calculate A
            // element to m/(t + beta)
            let beta = rngs[0];
            let A_i: HashMap<usize, Fr> = m_i.iter().map(|(i, y)| (*i, Fr::from(*y as u32) * (model.raw[*i] + beta).inverse().unwrap())).collect();
            // lagrange basis of value
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
            // msm basis with A(x)
            let A_x = util::msm::<G1Projective>(&temp, &temp2);
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (Q_i_x_1[*i], *y)).unzip();
            let A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
            let A_zero = srs.X1P[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>());
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_0_x_1[*i], *y)).unzip();
            let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);

            let mut rng2 = StdRng::from_entropy();
            let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
            let commits = vec![m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC];
            let mut commits: Vec<G1Projective> = commits.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
            let mut C = vec![
              -(srs.X1P[N] - srs.X1P[0]) * r[2] + model.g1 * r[1] + A_x * model.r + (srs.Y1P * model.r * r[1]) + srs.X1P[0] * (r[1] * beta - r[0]),
              -srs.X1P[1] * r[4] + srs.X1P[0] * (r[1] - r[3]),
              -srs.X1P[1] * r[7] + srs.X1P[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(*n as u32).inverse().unwrap()),
              -srs.X1P[0] * r[8] + srs.X1P[N - *n] * r[5],
            ];
            let proof_2 = vec![r[6], r[5], Fr::zero()];
            commits.append(&mut C);
            (commits, proof_2)
          } else {
            (proof_0.clone(), proof_2.clone())
          };

          let zeta = if rngs.len() == 1 { Fr::rand(rng) } else { rngs[1] };
          let mu = if rngs.len() == 1 { Fr::rand(rng) } else { rngs[2] };
          let mu_pow = if rngs.len() == 1 { mu } else { rngs[3] * mu };
          let q1_poly: DensePolynomial<Fr> = &polys[3] + &input.poly.mul(mu_pow);
          let f_z = input.poly.evaluate(&zeta);
          proof_2[2] = proof_2[2] + input.r * mu_pow;
          let mut evals = proof_2.clone();
          evals.push(f_z);
          let new_rngs = vec![rngs[0], zeta, mu, mu_pow];
          let mut new_polys = polys[..3].to_vec();
          new_polys.push(q1_poly);
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ(*n, m_i.clone(), *T_x_2, new_polys, new_rngs, proof, evals),
          )
        } else {
        }
      }
      _ => {}
    }
  }

  fn batch_prove(&self, srs: &SRS, batch_prove_values: &BatchProveStateValues, _rng: &mut StdRng) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    if let BatchProveStateValues::CQ(n, _, T_x_2, polys, rngs, proof_0, proof_2) = batch_prove_values {
      let zeta = rngs[1];
      let q1_poly = &polys[3];
      let q1_z = q1_poly.evaluate(&zeta);
      let q1_z_poly = DensePolynomial { coeffs: vec![q1_z] };
      let q1_V = DensePolynomial {
        coeffs: vec![-zeta, Fr::one()],
      };
      let W_poly = &q1_poly.sub(&q1_z_poly) / &q1_V;
      let W_x = util::msm::<G1Projective>(&srs.X1A, &W_poly.coeffs);

      let fz_prod: Fr = proof_2[3..].iter().product();
      let mut rng2 = StdRng::from_entropy();
      let r = &proof_2[..2];
      let r_sum = proof_2[2];
      let C5 = -(srs.X1P[*n] - srs.X1P[0]) * r[0] + srs.X1P[0] * fz_prod * r[1];
      let r = Fr::rand(&mut rng2);
      let C6 = -(srs.X1P[1] - srs.X1P[0] * zeta) * r + srs.X1P[0] * r_sum;

      let mut proof = proof_0.clone();
      proof.append(&mut vec![W_x, C5, C6]);
      (proof, vec![*T_x_2], proof_2.clone())
    } else {
      (vec![], vec![], vec![])
    }
  }

  fn batch_verify_first(
    &self,
    batch_verify_state: &mut BatchVerifyState,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) {
    let input = inputs[0].first().unwrap();
    let model = model.first().unwrap();
    let key = format!("{:?}_{}", self, input.len);
    let mut guard = batch_verify_state.lock().unwrap();
    match guard.get_mut(&key) {
      Some(value) => {
        let (_, BatchVerifyStateValues::CQ(n, N, _, g1s)) = value;
        let mut new_g1s = g1s.clone();
        new_g1s.push(input.g1);
        *value = (Box::new(self.clone()), BatchVerifyStateValues::CQ(*n, *N, model.g1, new_g1s))
      }
      _ => {
        let N = model.len;
        guard.insert(
          key,
          (Box::new(self.clone()), BatchVerifyStateValues::CQ(input.len, N, model.g1, vec![input.g1])),
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
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, W_x, C5, C6] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let [T_x_2] = proof.1[..] else { panic!("Wrong proof format") };
    let f_zs = proof.2;

    let beta = Fr::rand(rng);
    let zeta = Fr::rand(rng);
    let mu = Fr::rand(rng);

    let BatchVerifyStateValues::CQ(n, N, model_g1, f_xs) = batch_verify_values;

    let zetas = util::calc_pow(zeta, f_zs.len());
    let mus = util::calc_pow(mu, f_zs.len());
    let fz_sum: Fr = f_zs.iter().enumerate().map(|(i, x)| *x * mus[i]).sum();
    let fz_prod = f_zs.iter().product();
    let diff: Fr = f_zs.iter().map(|x| &fz_prod / x).sum();
    let f_sum: G1Projective = f_xs.iter().enumerate().map(|(i, x)| (*x + srs.X1A[0] * beta) * mus[i]).sum();

    // Check A(x) (A_i = m_i/(t_i+beta))
    checks.push(vec![
      (A_x, T_x_2),
      ((A_x * beta - m_x).into(), srs.X2A[0]),
      (-A_Q_x, (srs.X2A[*N] - srs.X2A[0]).into()),
      (-C1, srs.Y2A),
    ]);

    // Check T_x_2 is the G2 equivalent of the model
    checks.push(vec![(*model_g1, srs.X2A[0]), (srs.X1A[0], -T_x_2)]);

    // Check A(x) - A(0) is divisible by x
    checks.push(vec![((A_x - A_zero).into(), srs.X2A[0]), (-A_zero_div, srs.X2A[1]), (-C2, srs.Y2A)]);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0: G1Affine = (A_zero * (Fr::from(*N as u32) * Fr::from(*n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    checks.push(vec![((B_x - B_0).into(), srs.X2A[0]), (-B_zero_div, srs.X2A[1]), (-C3, srs.Y2A)]);

    // Degree check B
    checks.push(vec![(B_x, srs.X2A[N - n]), (-B_DC, srs.X2A[0]), (-C4, srs.Y2A)]);

    // Check B(x) (B_i = sum ())
    checks.push(vec![
      (
        (B_x * fz_prod - srs.X1A[0] * diff - B_Q_x * (zetas[n - 1] - Fr::one())).into(),
        srs.X2A[0],
      ),
      (-C5, srs.Y2A),
    ]);

    // Check W(x) (W(x) = (F(x) - F(zeta)) / (x - zeta))
    // where F(x) is RLC'd input polynomials
    checks.push(vec![
      ((-W_x, (srs.X2P[1] - srs.X2A[0] * zeta).into())),
      ((f_sum - srs.X1A[0] * fz_sum).into(), srs.X2A[0]),
      (-C6, srs.Y2A),
    ]);

    checks
  }
}
