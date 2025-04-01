#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, BasicBlockForTest, CacheValues, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::basic_block::cq::{cq_acc_clean, cq_acc_decide, cq_acc_finalize, CQLayoutHelper};
use crate::util::{self, acc_proof_to_acc, acc_to_acc_proof, get_cq_N, AccHolder, AccProofLayout};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{
  ops::{Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::ArrayD;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

impl AccProofLayout for CQ2BasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    CQLayoutHelper::acc_g1_num(is_prover)
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    CQLayoutHelper::acc_g2_num()
  }
  fn acc_fr_num(&self, is_prover: bool) -> usize {
    CQLayoutHelper::acc_fr_num(is_prover)
  }
  fn err_g1_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_g1_nums()
  }
  fn err_g2_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_g2_nums()
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_fr_nums()
  }
  fn err_gt_nums(&self) -> Vec<usize> {
    CQLayoutHelper::err_gt_nums()
  }
  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective> {
    CQLayoutHelper::prover_proof_to_acc(proof)
  }
  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine> {
    CQLayoutHelper::verifier_proof_to_acc(proof)
  }
  fn mira_prove(
    &self,
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective> {
    CQLayoutHelper::mira_prove(srs, acc_1, acc_2, rng)
  }
  fn mira_verify(
    &self,
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    CQLayoutHelper::mira_verify(acc_1, acc_2, new_acc, rng)
  }
}

#[derive(Debug)]
pub struct CQ2BasicBlock {
  pub n: usize,
  pub setup: Option<(Box<dyn BasicBlock>, i128, usize)>,
}

impl BasicBlock for CQ2BasicBlock {
  fn genModel(&self) -> ArrayD<Fr> {
    util::gen_cq_table(
      &(self.setup.as_ref().unwrap().0),
      self.setup.as_ref().unwrap().1,
      self.setup.as_ref().unwrap().2, // N
    )
  }

  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    if model.ndim() != 2 {
      return Ok(vec![]);
    }
    assert!(inputs.len() == 2);
    let n = inputs[0].len();
    assert!(n == self.n, "self.n is not equal to n, which is {}", inputs[0].len());
    if self.setup.is_some() && !self.setup.as_ref().unwrap().0.is::<BasicBlockForTest>() {
      for x in inputs[0].iter().zip(inputs[1].iter()) {
        let temp = (*x.0, *x.1);
        let x_0_int = util::fr_to_int(temp.0);
        let low = self.setup.as_ref().unwrap().1;
        let high = low + self.setup.as_ref().unwrap().2 as i128;
        if x_0_int < low || x_0_int >= high {
          let temp_ints = (util::fr_to_int(temp.0), util::fr_to_int(temp.1));
          println!("{:?}, {:?}", temp_ints, temp);
          return Err(util::CQOutOfRangeError { input: temp_ints.0 });
        }
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

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    assert!(inputs.len() == 2 && inputs[0].len() == 1 && inputs[1].len() == 1);
    let N = model[0].raw.len();
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    assert!(inputs[0].raw.len() == inputs[1].raw.len());
    let n = inputs[0].raw.len();
    assert!(n == self.n);
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let alpha = Fr::rand(rng);
    let agg_input: Vec<_> = inputs[0].raw.iter().zip(inputs[1].raw.iter()).map(|(x, y)| *x + *y * alpha).collect();
    let mut agg_input = Data::new(srs, &agg_input); // Unnecessary msm
    agg_input.r = inputs[0].r + inputs[1].r * alpha;
    let agg_model_g1 = model[0].g1 + model[1].g1 * alpha;
    let agg_model_r = model[0].r + model[1].r * alpha;

    // gen(N, t):
    let Q_i_x_1_A = &setup.0[..N];
    let Q_i_x_1_B = &setup.0[N..2 * N];
    let L_i_x_1 = &setup.0[2 * N..3 * N];
    let L_i_0_x_1 = &setup.0[3 * N..];

    let m_i = {
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

      // Calculate m
      let mut m_i = HashMap::new();
      for x in inputs[0].raw.iter().zip(inputs[1].raw.iter()) {
        let temp = (*x.0, *x.1);
        if !table_dict.contains_key(&temp) {
          println!("{:?}", temp);
        }
        m_i.entry(table_dict.get(&temp).unwrap().clone()).and_modify(|y| *y += 1).or_insert(1);
      }
      m_i
    };
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = m_i.iter().map(|(i, y)| (L_i_x_1[*i], Fr::from(*y as u32))).unzip();
    let m_x = util::msm::<G1Projective>(&temp, &temp2);

    let beta = Fr::rand(rng);

    // Calculate A
    let A_i: HashMap<usize, Fr> = m_i
      .iter()
      .map(|(i, y)| {
        (
          *i,
          Fr::from(*y as u32) * (model[0].raw[*i] + model[1].raw[*i] * alpha + beta).inverse().unwrap(),
        )
      })
      .collect();
    let (temp, temp2): (Vec<G1Affine>, Vec<Fr>) = A_i.iter().map(|(i, y)| (L_i_x_1[*i], *y)).unzip();
    let A_x = util::msm::<G1Projective>(&temp, &temp2);
    let temp: Vec<G1Projective> = A_i.iter().map(|(i, _)| Q_i_x_1_A[*i] + Q_i_x_1_B[*i] * alpha).collect();
    let temp: Vec<G1Affine> = temp.iter().map(|x| (*x).into()).collect();
    let A_Q_x = util::msm::<G1Projective>(&temp, &temp2);
    let A_zero = srs.X1P[0] * (Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_, y)| *y).sum::<Fr>());
    let temp: Vec<G1Affine> = A_i.iter().map(|(i, _)| L_i_0_x_1[*i]).collect();
    let A_zero_div = util::msm::<G1Projective>(&temp, &temp2);

    // Calculate B
    let B_i: Vec<Fr> = agg_input.raw.iter().map(|x| (*x + beta).inverse().unwrap()).collect();
    let B_poly = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_Q_poly = B_poly
      .mul(&(agg_input.poly.clone() + (DensePolynomial::from_coefficients_vec(vec![beta]))))
      .sub(&DensePolynomial::from_coefficients_vec(vec![Fr::one()]))
      .divide_by_vanishing_poly(domain_n)
      .unwrap()
      .0;
    let B_x = util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs);
    let B_Q_x = util::msm::<G1Projective>(&srs.X1A, &B_Q_poly.coeffs);
    let B_zero_div = if B_poly.is_zero() {
      G1Projective::zero()
    } else {
      util::msm::<G1Projective>(&srs.X1A, &B_poly.coeffs[1..])
    };
    let B_DC = util::msm::<G1Projective>(&srs.X1A[N - n..], &B_poly.coeffs);

    let f_x_2 = util::msm::<G2Projective>(&srs.X2A, &agg_input.poly.coeffs) + srs.Y2P * agg_input.r;

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..9).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Projective> = vec![m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC];
    let mut proof: Vec<G1Projective> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    let part_C1 = -(srs.X1P[N] - srs.X1P[0]) * r[2] - srs.X1P[0] * r[0];
    let part_C3 = -(srs.X1P[n] - srs.X1P[0]) * r[6];
    let mut C = vec![
      part_C1 + agg_model_g1 * r[1] + A_x * agg_model_r + (srs.Y1P * agg_model_r * r[1]) + srs.X1P[0] * r[1] * beta,
      -srs.X1P[1] * r[4] + srs.X1P[0] * (r[1] - r[3]),
      part_C3 + agg_input.g1 * r[5] + B_x * agg_input.r + (srs.Y1P * agg_input.r * r[5]) + srs.X1P[0] * (r[5] * beta),
      -srs.X1P[1] * r[7] + srs.X1P[0] * (r[5] - r[3] * Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap()),
      -srs.X1P[0] * r[8] + srs.X1P[N - n] * r[5],
    ];
    proof.append(&mut C);
    let mut fr: Vec<Fr> = vec![beta];

    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![
        agg_model_g1 + srs.Y1P * agg_model_r,
        agg_input.g1 + srs.Y1P * agg_input.r,
        part_C1,
        part_C3,
        agg_model_g1,
        agg_input.g1,
        A_x,
        B_x,
      ];

      proof.append(&mut additional_g1_for_acc);
      fr.append(&mut vec![agg_model_r, agg_input.r, r[1], r[5]]);
    }

    return (proof, vec![(setup.1[0] + setup.1[1] * alpha).into(), f_x_2], fr);
  }

  #[cfg(not(feature = "fold"))]
  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    let N = model[0].len;
    let n = inputs[0].len;
    let alpha = Fr::rand(rng);
    let [m_x, A_x, A_Q_x, A_zero, A_zero_div, B_x, B_Q_x, B_zero_div, B_DC, C1, C2, C3, C4, C5] = proof.0[..14] else {
      panic!("Wrong proof format")
    };
    let [T_x_2, f_x_2] = proof.1[..] else { panic!("Wrong proof format") };
    let agg_input = (inputs[0].g1 + (inputs[1].g1 * alpha)).into();
    let agg_model = (model[0].g1 + (model[1].g1 * alpha)).into();

    let beta = Fr::rand(rng);

    // Check A(x) (A_i = m_i/(t_i+beta))
    checks.push(vec![
      (A_x, T_x_2),
      ((A_x * beta - m_x).into(), srs.X2A[0]),
      (-A_Q_x, (srs.X2A[N] - srs.X2A[0]).into()),
      (-C1, srs.Y2A),
    ]);

    // Check T_x_2 is the G2 equivalent of the model
    checks.push(vec![(agg_model, srs.X2A[0]), (srs.X1A[0], -T_x_2)]);

    // Check A(x) - A(0) is divisible by x
    checks.push(vec![((A_x - A_zero).into(), srs.X2A[0]), (-A_zero_div, srs.X2A[1]), (-C2, srs.Y2A)]);

    // Check B(x) (B_i = 1/(f_i+beta))
    checks.push(vec![
      (B_x, f_x_2),
      ((B_x * beta - srs.X1A[0]).into(), srs.X2A[0]),
      (-B_Q_x, (srs.X2A[n] - srs.X2A[0]).into()),
      (-C3, srs.Y2A),
    ]);

    // Check f_x_2 is the G2 equivalent of the input
    checks.push(vec![(agg_input, srs.X2A[0]), (srs.X1A[0], -f_x_2)]);

    // Assume B(0) = A(0)*N/n (which assumes ∑A=∑B)
    let B_0: G1Affine = (A_zero * (Fr::from(N as u32) * Fr::from(n as u32).inverse().unwrap())).into();

    // Check B(x) - B(0) is divisible by x
    checks.push(vec![((B_x - B_0).into(), srs.X2A[0]), (-B_zero_div, srs.X2A[1]), (-C4, srs.Y2A)]);

    // Degree check B
    checks.push(vec![(B_x, srs.X2A[N - n]), (-B_DC, srs.X2A[0]), (-C5, srs.Y2A)]);

    checks
  }

  #[cfg(feature = "fold")]
  fn verify(
    &self,
    _srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let alpha = Fr::rand(rng);
    let inputs = vec![inputs[0].first().unwrap(), inputs[1].first().unwrap()];
    let agg_input: G1Affine = (inputs[0].g1 + (inputs[1].g1 * alpha)).into();
    let agg_model: G1Affine = (model[0].g1 + (model[1].g1 * alpha)).into();

    let beta = Fr::rand(rng);
    let mut result = beta == proof.2[0];
    result &= agg_model == proof.0[proof.0.len() - 2];
    result &= agg_input == proof.0[proof.0.len() - 1];
    //assert!(result, "acc_proof for cq2 is not valid");
    vec![]
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let proof = self.prover_proof_to_acc(proof);
    if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
      return acc_to_acc_proof(proof);
    }
    let acc_proof = acc_proof_to_acc(self, acc_proof, true);
    acc_to_acc_proof(self.mira_prove(srs, acc_proof, proof, rng))
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (
      &Vec<G1Projective>,
      &Vec<G2Projective>,
      &Vec<Fr>,
      &Vec<PairingOutput<Bn<ark_bn254::Config>>>,
    ),
  ) -> (
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>),
    (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) {
    cq_acc_clean(self, srs, proof, acc_proof)
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    _outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);

    let mut result = true;

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      return Some(result);
    }
    let proof = self.verifier_proof_to_acc(proof);
    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);
    result &= self.mira_verify(prev_acc_holder, proof, acc_holder, rng).unwrap();
    Some(result)
  }

  fn acc_decide(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> Vec<(PairingCheck, PairingOutput<Bn<ark_bn254::Config>>)> {
    let N = self.setup.as_ref().unwrap().2;
    let n = self.n;
    cq_acc_decide(self, srs, acc_proof, N, n)
  }

  fn acc_finalize(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let N = self.setup.as_ref().unwrap().2;
    let n = self.n;
    cq_acc_finalize(self, srs, acc_proof, N, n)
  }
}
