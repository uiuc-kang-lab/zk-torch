#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, BatchProveStateValues, BatchVerifyStateValues, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::basic_block::*;
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
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum CQ2BasicBlockOps {
  Ceil(usize, usize),
  ChangeSF(usize, usize),
  Clip(f32, f32),
  Cos(usize, usize),
  DivConst(f32),
  Erf(usize, usize),
  Exp(usize, usize),
  GeLU(usize, usize),
  Log(usize, usize),
  ModConst(u32),
  Reciprocal(usize, usize),
  ReLU(usize, usize),
  Sigmoid(usize, usize),
  Sin(usize, usize),
  Sqrt(usize, usize),
  Tan(usize, usize),
  Tanh(usize, usize),
}

fn enum_to_bb(bb: &CQ2BasicBlockOps) -> Box<dyn BasicBlock> {
  match bb {
    CQ2BasicBlockOps::Ceil(input_SF, output_SF) => Box::new(CeilBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::ChangeSF(input_SF, output_SF) => Box::new(ChangeSFBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Cos(input_SF, output_SF) => Box::new(CosBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Clip(min, max) => Box::new(ClipBasicBlock { min: *min, max: *max }),
    CQ2BasicBlockOps::DivConst(c) => Box::new(DivConstBasicBlock { c: *c }),
    CQ2BasicBlockOps::Erf(input_SF, output_SF) => Box::new(ErfBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Exp(input_SF, output_SF) => Box::new(ExpBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::GeLU(input_SF, output_SF) => Box::new(GeLUBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Log(input_SF, output_SF) => Box::new(LogBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::ModConst(c) => Box::new(ModConstBasicBlock { c: *c }),
    CQ2BasicBlockOps::Reciprocal(input_SF, output_SF) => Box::new(ReciprocalBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::ReLU(input_SF, output_SF) => Box::new(ReLUBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Sigmoid(input_SF, output_SF) => Box::new(SigmoidBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Sin(input_SF, output_SF) => Box::new(SinBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Sqrt(input_SF, output_SF) => Box::new(SqrtBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Tan(input_SF, output_SF) => Box::new(TanBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
    CQ2BasicBlockOps::Tanh(input_SF, output_SF) => Box::new(TanhBasicBlock {
      input_SF: *input_SF,
      output_SF: *output_SF,
    }),
  }
}

#[derive(Debug, Clone)]
pub struct CQ2BasicBlock {
  pub op: CQ2BasicBlockOps,
  pub offset: i128,
  pub size: usize,
}

impl BasicBlock for CQ2BasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    util::gen_cq_table(&enum_to_bb(&self.op), self.offset, self.size)
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    if model.ndim() != 2 {
      return Ok(vec![]);
    }
    assert!(inputs.len() == 2);
    for x in inputs[0].iter().zip(inputs[1].iter()) {
      let temp = (*x.0, *x.1);
      let x_0_int = util::fr_to_int(temp.0);
      let low = self.offset;
      let high = low + self.size as i128;
      if x_0_int < low || x_0_int >= high {
        let temp_ints = (util::fr_to_int(temp.0), util::fr_to_int(temp.1));
        println!("{:?}, {:?}", temp_ints, temp);
        return Err(util::CQOutOfRangeError { input: temp_ints.0 });
      }
    }

    Ok(vec![])
  }

  #[cfg(not(feature = "mock_prove"))]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    assert!(model.ndim() == 1 && model.len() == 2);
    let N = model[0].raw.len();
    let domain_2N = GeneralEvaluationDomain::<Fr>::new(2 * N).unwrap();
    let domain_N = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let mut setup = vec![];
    let mut setup2 = vec![];
    for i in 0..2 {
      setup2.push(util::msm::<G2Projective>(&srs.X2A, &model[i].poly.coeffs) + srs.Y2P * model[i].r);
      let mut temp = model[i].poly.coeffs[1..].to_vec();
      temp.resize(N * 2 - 1, Fr::zero());
      let mut temp2 = srs.X1P[..N].to_vec();
      temp2.reverse();
      let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
      util::fft_in_place(domain_N, &mut Q_i_x_1);
      let temp = Fr::from(N as u32).inverse().unwrap();
      let temp2 = domain_N.group_gen_inv().pow(&[(N - 1) as u64]);
      let scalars = (0..N).into_par_iter().map(|i| temp * temp2.pow(&[i as u64])).collect();
      util::ssm_g1_in_place(&mut Q_i_x_1, &scalars);

      setup.extend(Q_i_x_1);
    }

    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain_N, &mut L_i_x_1);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let scalars = (0..N).into_par_iter().map(|i| domain_N.group_gen_inv().pow(&[i as u64])).collect();
    util::ssm_g1_in_place(&mut L_i_0_x_1, &scalars);
    let temp = srs.X1P[N - 1] * Fr::from(N as u64).inverse().unwrap();
    L_i_0_x_1.par_iter_mut().for_each(|x| *x -= temp);

    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, setup2, Vec::new());
  }

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    assert!(model.ndim() == 1 && model.len() == 2);
    let N = model[0].raw.len();
    let mut setup = vec![];
    let mut setup2 = vec![];
    for i in 0..2 {
      setup2.push(srs.X2P[i]);
    }
    let Q_i_x_1_A = srs.X1P[..N].to_vec();
    let Q_i_x_1_B = srs.X1P[..N].to_vec();
    let L_i_x_1 = srs.X1P[..N].to_vec();
    let L_i_0_x_1 = srs.X1P[..N].to_vec();

    setup.extend(Q_i_x_1_A);
    setup.extend(Q_i_x_1_B);
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup, setup2, Vec::new());
  }

  fn batch_prove_first(
    &self,
    _srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    batch_prove_state: &mut BatchProveState,
    batch_counters: &mut BatchCounters,
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) {
    assert!(inputs.len() == 2 && inputs[0].len() == 1 && inputs[1].len() == 1);
    let N = model[0].raw.len();
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    assert!(inputs[0].raw.len() == inputs[1].raw.len());
    let n = inputs[0].raw.len();

    let tag = format!("{:?}_{}", self, inputs[0].raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut cache = cache.lock().unwrap();
    let CacheValues::CQ2TableDict(table_dict) =
      cache.entry(format!("cq2_table_dict_{:p}", self)).or_insert_with(|| CacheValues::CQ2TableDict(HashMap::new()))
    else {
      panic!("Cache type error")
    };
    if table_dict.len() == 0 {
      for i in 0..N {
        table_dict.insert((model[0].raw[i], model[1].raw[i]), i);
      }
    }
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ2(n, m_i, T_x_2, _, _, _, _)) = value {
          for x in inputs[0].raw.iter().zip(inputs[1].raw.iter()) {
            let temp = (*x.0, *x.1);
            if !table_dict.contains_key(&temp) {
              println!("{:?}", temp);
            }
            m_i.borrow_mut().entry(table_dict.get(&temp).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
          }
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ2(
              *n,
              RefCell::clone(m_i),
              *T_x_2,
              RefCell::new(vec![DensePolynomial::from_coefficients_vec(vec![Fr::zero()])]),
              vec![],
              vec![],
              RefCell::new(vec![]),
            ),
          )
        } else {
        }
      }
      _ => {
        let mut m_i = HashMap::new();
        for x in inputs[0].raw.iter().zip(inputs[1].raw.iter()) {
          let temp = (*x.0, *x.1);
          if !table_dict.contains_key(&temp) {
            println!("{:?}", temp);
            m_i.entry(table_dict.get(&temp).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
          }
        }
        state_mut_ref.insert(
          key.to_string(),
          (
            Box::new(self.clone()),
            BatchProveStateValues::CQ2(
              n,
              RefCell::new(m_i),
              setup.1[0].into(),
              RefCell::new(vec![DensePolynomial::from_coefficients_vec(vec![Fr::zero()])]),
              vec![],
              vec![],
              RefCell::new(vec![]),
            ),
          ),
        );
      }
    }
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
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    let n = inputs[0].raw.len();
    let N = model.raw.len();
    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    let tag = format!("{:?}_{}", self, inputs[0].raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ2(n, m_i, T_x_2, polys, rngs, _, _)) = value {
          let alpha = if rngs.len() == 0 { Fr::rand(rng) } else { rngs[0] };
          let beta = if rngs.len() == 0 { Fr::rand(rng) } else { rngs[1] };
          let agg_input: Vec<_> = inputs[0].raw.iter().zip(inputs[1].raw.iter()).map(|(x, y)| *x + *y * alpha).collect();

          let B_i: Vec<Fr> = agg_input.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
          let agg_input_poly = DensePolynomial::from_coefficients_vec(domain_n.ifft(&agg_input));
          let f_poly = agg_input_poly + DensePolynomial::from_coefficients_vec(vec![beta]);
          let agg_B = polys.borrow()[0].clone();
          polys.borrow_mut()[0] = &agg_B + &Evaluations::from_vec_and_domain(B_i, domain_n).interpolate();
          polys.borrow_mut().push(f_poly);
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ2(
              *n,
              RefCell::clone(m_i),
              *T_x_2,
              RefCell::clone(polys),
              vec![alpha, beta],
              vec![],
              RefCell::new(vec![]),
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
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    let n = inputs[0].raw.len();
    let N = model[0].raw.len();

    // gen(N, t):
    let Q_i_x_1_A = &setup.0[..N];
    let Q_i_x_1_B = &setup.0[N..2 * N];
    let L_i_x_1 = &setup.0[2 * N..3 * N];
    let L_i_0_x_1 = &setup.0[3 * N..];

    assert!(n <= N);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let tag = format!("{:?}_{}", self, inputs[0].raw.len());
    let key = util::update_batch_counters(batch_counters, &tag, 8);
    let mut state_mut_ref = batch_prove_state.borrow_mut();
    match state_mut_ref.get_mut(&key) {
      Some(value) => {
        if let (_, BatchProveStateValues::CQ2(n, m_i, T_x_2, polys, rngs, proof_0, proof_2)) = value {
          let alpha = rngs[0];
          let (proof, proof_2) = if proof_0.len() == 0 {
            let poly_ref = polys.borrow();
            let B_poly = &poly_ref[0];
            let beta = rngs[0];
            let fs: Vec<_> = poly_ref[1..].iter().map(|x| x + &DensePolynomial::from_coefficients_vec(vec![beta])).collect();
            let f_prod = util::mul_polys(&fs);
            let diffs: Vec<_> = poly_ref[1..].iter().map(|x| &f_prod / x).collect();
            let diff = diffs.iter().fold(DensePolynomial::zero(), |acc, x| acc + x.clone());

            let agg_model_g1 = model[0].g1 + model[1].g1 * alpha;
            let agg_model_r = model[0].r + model[1].r * alpha;

            let B_Q_poly = B_poly.mul(&f_prod).sub(&diff).divide_by_vanishing_poly(domain_n).unwrap().0;
            let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
            let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
            let B_zero_div = if B_poly.is_zero() {
              G1Projective::zero()
            } else {
              util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..])
            };
            let B_DC = util::msm::<G1Projective>(&srs.X1A[N - *n..], &B_poly.coeffs);

            let m_ref = m_i.borrow();
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_ref.iter().map(|(i, y)| (L_i_x_1[*i], Fr::from(*y as u32))).unzip();
            let m_x = util::msm::<G1Projective>(&temp, &temp2);

            // Calculate A
            // element to m/(t + beta)
            let A_i: HashMap<usize, Fr> = m_ref
              .iter()
              .map(|(i, y)| {
                (
                  *i,
                  Fr::from(*y as u32) * (model[0].raw[*i] + model[1].raw[*i] * alpha + beta).inverse().unwrap(),
                )
              })
              .collect();
            // lagrange basis of value
            let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
            // msm basis with A(x)
            let A_x = util::msm::<G1Projective>(&temp, &temp2);
            let temp: Vec<G1Projective> = A_i.iter().map(|(i, _)| Q_i_x_1_A[*i] + Q_i_x_1_B[*i] * alpha).collect();
            let temp: Vec<G1Affine> = temp.iter().map(|x| (*x).into()).collect();
            let A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
            let A_zero = srs.X1P[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>());
            let temp: Vec<G1Affine> = A_i.iter().map(|(i, _)| L_i_0_x_1[*i]).collect();
            let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);

            let mut rng2 = StdRng::from_entropy();
            let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
            let mut proof: Vec<_> = vec![m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC]
              .iter()
              .enumerate()
              .map(|(i, x)| (*x) + srs.Y1P * r[i])
              .collect();
            let mut C = vec![
              -(srs.X1P[N] - srs.X1P[0]) * r[2]
                + agg_model_g1 * r[1]
                + A_x * agg_model_r
                + (srs.Y1P * agg_model_r * r[1])
                + srs.X1P[0] * (r[1] * beta - r[0]),
              -srs.X1P[1] * r[4] + srs.X1P[0] * (r[1] - r[3]),
              -srs.X1P[1] * r[7] + srs.X1P[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(*n as u32).inverse().unwrap()),
              -srs.X1P[0] * r[8] + srs.X1P[N - *n] * r[5],
            ];
            proof.append(&mut C);
            (proof, &mut RefCell::new(vec![r[6], r[5], Fr::zero()]))
          } else {
            (proof_0.clone(), proof_2)
          };

          let zeta = if rngs.len() == 2 { Fr::rand(rng) } else { rngs[2] };
          let mu = if rngs.len() == 2 { Fr::rand(rng) } else { rngs[3] };
          let mu_pow = if rngs.len() == 2 { mu } else { rngs[4] * mu };
          let agg_input: Vec<_> = inputs[0].raw.iter().zip(inputs[1].raw.iter()).map(|(x, y)| *x + *y * alpha).collect();
          let agg_input_r = inputs[0].r + inputs[1].r * alpha;
          let agg_input_poly = DensePolynomial::from_coefficients_vec(domain_n.ifft(&agg_input));
          let poly_ref = polys.borrow();
          let q1_poly: DensePolynomial<Fr> = if poly_ref.len() > 0 {
            agg_input_poly.mul(mu_pow)
          } else {
            &poly_ref[0] + &agg_input_poly.mul(mu_pow)
          };
          drop(poly_ref);
          let f_z = agg_input_poly.evaluate(&zeta);
          let agg_r = proof_2.borrow()[2];
          proof_2.borrow_mut()[2] = agg_r + agg_input_r * mu_pow;
          proof_2.borrow_mut().push(f_z);

          let new_rngs = vec![rngs[0], rngs[1], zeta, mu, mu_pow];
          // let mut new_polys = polys[..3].to_vec();
          // new_polys.push(q1_poly);
          *value = (
            Box::new(self.clone()),
            BatchProveStateValues::CQ2(
              *n,
              RefCell::clone(m_i),
              *T_x_2,
              RefCell::new(vec![q1_poly]),
              new_rngs,
              proof.clone(),
              RefCell::clone(proof_2),
            ),
          )
        } else {
        }
      }
      _ => {}
    }
  }

  fn batch_prove(&self, srs: &SRS, batch_prove_values: &BatchProveStateValues, _rng: &mut StdRng) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    if let BatchProveStateValues::CQ2(n, _, T_x_2, polys, rngs, proof_0, proof_2) = batch_prove_values {
      let zeta = rngs[2];
      let q1_poly = &polys.borrow()[0];
      let q1_z = q1_poly.evaluate(&zeta);
      let q1_z_poly = DensePolynomial { coeffs: vec![q1_z] };
      let q1_V = DensePolynomial {
        coeffs: vec![-zeta, Fr::one()],
      };
      let W_poly = &q1_poly.sub(&q1_z_poly) / &q1_V;
      let W_x = util::msm::<G1Projective>(&srs.X1A, &W_poly.coeffs);

      let p_ref = proof_2.borrow();
      let fz_prod: Fr = p_ref[3..].iter().product();
      let mut rng2 = StdRng::from_entropy();
      let r = &p_ref[..2];
      let r_sum = p_ref[2];
      let C5 = -(srs.X1P[*n] - srs.X1P[0]) * r[0] + srs.X1P[0] * fz_prod * r[1];
      let r = Fr::rand(&mut rng2);
      let C6 = -(srs.X1P[1] - srs.X1P[0] * zeta) * r + srs.X1P[0] * r_sum;

      let mut proof = proof_0.clone();
      proof.append(&mut vec![W_x, C5, C6]);
      (proof, vec![*T_x_2], p_ref[3..].to_vec())
    } else {
      (vec![], vec![], vec![])
    }
  }

  // fn batch_verify_first(
  //   &self,
  //   batch_verify_state: &mut BatchVerifyState,
  //   model: &ArrayD<DataEnc>,
  //   inputs: &Vec<&ArrayD<DataEnc>>,
  //   _outputs: &Vec<&ArrayD<DataEnc>>,
  //   _rng: &mut StdRng,
  //   _cache: ProveVerifyCache,
  // ) {
  //   let input = inputs[0].first().unwrap();
  //   let model = model.first().unwrap();
  //   let key = format!("{:?}_{}", self, input.len);
  //   let mut guard = batch_verify_state.lock().unwrap();
  //   match guard.get_mut(&key) {
  //     Some(value) => {
  //       let (_, BatchVerifyStateValues::CQ(n, N, _, g1s)) = value;
  //       let mut new_g1s = g1s.clone();
  //       new_g1s.push(input.g1);
  //       *value = (Box::new(self.clone()), BatchVerifyStateValues::CQ(*n, *N, model.g1, new_g1s))
  //     }
  //     _ => {
  //       let N = model.len;
  //       guard.insert(
  //         key,
  //         (Box::new(self.clone()), BatchVerifyStateValues::CQ(input.len, N, model.g1, vec![input.g1])),
  //       );
  //     }
  //   };
  // }

  // fn batch_verify(
  //   &self,
  //   srs: &SRS,
  //   proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
  //   batch_verify_values: &BatchVerifyStateValues,
  //   rng: &mut StdRng,
  // ) -> Vec<PairingCheck> {
  //   let mut checks = Vec::new();
  //   let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, W_x, C5, C6] = proof.0[..] else {
  //     panic!("Wrong proof format")
  //   };
  //   let [T_x_2] = proof.1[..] else { panic!("Wrong proof format") };
  //   let f_zs = proof.2;

  //   let beta = Fr::rand(rng);
  //   let zeta = Fr::rand(rng);
  //   let mu = Fr::rand(rng);

  //   let BatchVerifyStateValues::CQ(n, N, model_g1, f_xs) = batch_verify_values;

  //   let zetas = util::calc_pow(zeta, f_zs.len());
  //   let mus = util::calc_pow(mu, f_zs.len());
  //   let fz_sum: Fr = f_zs.iter().enumerate().map(|(i, x)| *x * mus[i]).sum();
  //   let fz_prod = f_zs.iter().product();
  //   let diff: Fr = f_zs.iter().map(|x| &fz_prod / x).sum();
  //   let f_sum: G1Projective = f_xs.iter().enumerate().map(|(i, x)| (*x + srs.X1A[0] * beta) * mus[i]).sum();

  //   // Check A(x) (A_i = m_i/(t_i+beta))
  //   checks.push(vec![
  //     (A_x, T_x_2),
  //     ((A_x * beta - m_x).into(), srs.X2A[0]),
  //     (-A_Q_x, (srs.X2A[*N] - srs.X2A[0]).into()),
  //     (-C1, srs.Y2A),
  //   ]);

  //   // Check T_x_2 is the G2 equivalent of the model
  //   checks.push(vec![(*model_g1, srs.X2A[0]), (srs.X1A[0], -T_x_2)]);

  //   // Check A(x) - A(0) is divisible by x
  //   checks.push(vec![((A_x - A_zero).into(), srs.X2A[0]), (-A_zero_div, srs.X2A[1]), (-C2, srs.Y2A)]);

  //   // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
  //   let B_0: G1Affine = (A_zero * (Fr::from(*N as u32) * Fr::from(*n as u32).inverse().unwrap())).into();

  //   // Check B(x) - B(0) is divisible by x
  //   checks.push(vec![((B_x - B_0).into(), srs.X2A[0]), (-B_zero_div, srs.X2A[1]), (-C3, srs.Y2A)]);

  //   // Degree check B
  //   checks.push(vec![(B_x, srs.X2A[N - n]), (-B_DC, srs.X2A[0]), (-C4, srs.Y2A)]);

  //   // Check B(x) (B_i = sum ())
  //   checks.push(vec![
  //     (
  //       (B_x * fz_prod - srs.X1A[0] * diff - B_Q_x * (zetas[n - 1] - Fr::one())).into(),
  //       srs.X2A[0],
  //     ),
  //     (-C5, srs.Y2A),
  //   ]);

  //   checks.push(vec![
  //     ((-W_x, (srs.X2P[1] - srs.X2A[0] * zeta).into())),
  //     ((f_sum - srs.X1A[0] * fz_sum).into(), srs.X2A[0]),
  //     (-C6, srs.Y2A),
  //   ]);

  //   checks
  // }
}
