#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::{
  util::{self, calc_pow},
  PairingCheck, ProveVerifyCache,
};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::Field;
use ark_poly::{
  evaluations::univariate::Evaluations, univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
};
use ark_serialize::CanonicalSerialize;
use ark_std::{
  ops::{Add, Mul, Sub},
  One, UniformRand, Zero,
};
use ndarray::{azip, indices, ArrayD, ArrayView, ArrayView1, ArrayViewD, Axis, Dim, Dimension, IxDyn, IxDynImpl, NdIndex, Shape, Zip};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::{
  cmp::{max, min},
  collections::HashMap,
  iter::{once, repeat},
};

fn flat_index(shape: &IxDyn, idx: &IxDyn, N: usize) -> usize {
  assert!(shape.ndim() == idx.ndim());
  let mut product = vec![];
  // If inputs and outputs do not have the same last dimension, then the one
  // with the smaller dimension will have had their polynomials constructed from
  // a smaller evaluation domain. This indexing enables the smaller dimension's
  // roots of unity evaluation values to line up to the larger one.
  let spacing = N / shape[shape.ndim() - 1];
  for d in 0..shape.ndim() {
    if d == 0 {
      product.push(spacing);
    } else {
      product.push(product[d - 1] * shape[shape.ndim() - d]);
    }
  }
  product = product.into_iter().rev().collect();
  idx.as_array_view().iter().enumerate().fold(0, |sum, (i, x)| *x * product[i] + sum)
}

// Construct S_sigma_j polynomial evals over N-th roots of unity
// which encodes the permutation function
// In a copy-constraint, the permutation should form a cycle for all elements
// that should be the same over inputs and outputs.
// If is_input, idxs is [(flat_idx of the input index, 0)]. Otherwise,
// idxs is [(flat_idx of the output, flat_idx of the permuted input idx)]
fn construct_ssig(idxs: &[(usize, usize)], N: usize, last_dim: usize, partitions: &HashMap<usize, Vec<usize>>, is_input: bool) -> Vec<Fr> {
  idxs
    .iter()
    .flat_map(|(idx, perm_idx)| {
      let inp_idx = if is_input { idx } else { perm_idx };
      let sigma = {
        let partition = partitions.get(&inp_idx).ok_or_else(|| format!("Key {:?} not found in the partition", inp_idx)).unwrap();
        if let Some(pos) = partition.iter().position(|x| *x == *idx) {
          Ok(partition[(pos + 1) % partition.len()])
        } else {
          Err(format!("Value {:?} not found in the list", *idx))
        }
      };
      let sigma = sigma.unwrap();
      let mut ssig = vec![Fr::from(sigma as i32)];
      // Permute each filler element to itself
      let mut padding: Vec<_> = (1..N / last_dim).map(|i| Fr::from((i + *idx) as i32)).collect();
      ssig.append(&mut padding);
      ssig
    })
    .collect()
}

// Permutation has the same shape as the output, and each index stores the index of the input array it equals to
#[derive(Debug)]
pub struct CopyConstraintBasicBlock {
  pub permutation: ArrayD<IxDyn>,
  pub input_dim: IxDyn,
  pub output_dim: IxDyn,
}

impl BasicBlock for CopyConstraintBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1);
    assert!(Dim(self.permutation.shape()) == self.output_dim);
    vec![ArrayD::from_shape_fn(self.permutation.shape(), |i| inputs[0][&self.permutation[i]])]
  }

  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let last_inp_dim = self.input_dim[self.input_dim.ndim() - 1];
    let last_outp_dim = self.output_dim[self.output_dim.ndim() - 1];
    let N = max(last_inp_dim, last_outp_dim);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let mut L_i_x_1 = srs.X1P[..N].to_vec();
    util::ifft_in_place(domain, &mut L_i_x_1);
    let mut L_i_x_2 = srs.X2P[..N].to_vec();
    util::ifft_in_place(domain, &mut L_i_x_2);

    // Indices of the output permutation elements
    // Offset output indices by the size of the input shape
    let mut input_dim = self.input_dim.as_array_view().to_vec().clone();
    input_dim[self.input_dim.ndim() - 1] = N;
    let offset: usize = input_dim.iter().product();
    let flat_outp_idxs = self.permutation.indexed_iter().map(|(i, _)| flat_index(&self.output_dim, &i, N) + offset).collect();
    let flat_outp_idxs = ArrayD::from_shape_vec(self.output_dim.clone(), flat_outp_idxs).unwrap();
    // Indices of the input which are in each position of the permutation
    let flat_perm_idxs = self.permutation.map(|i| flat_index(&self.input_dim, i, N));

    // Create partitions
    let mut partitions = HashMap::new();
    for i in indices(self.input_dim.clone()).into_iter() {
      let idx = flat_index(&self.input_dim, &i, N);
      partitions.entry(idx).or_insert_with(|| Vec::from([idx]));
    }
    for (i, _) in flat_outp_idxs.indexed_iter() {
      let out_idx = flat_outp_idxs[&i];
      let perm_idx = flat_perm_idxs[&i];
      partitions.entry(perm_idx).or_insert_with(|| Vec::from([perm_idx])).push(out_idx);
    }

    // Calculate S_ID
    let sid: Vec<_> = (0..N).map(|x| Fr::from(x as i32)).collect();
    let sid_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&sid));
    let sid_x = util::msm::<G1Projective>(&srs.X1A, &sid_poly.coeffs);

    // Calculate S_sigma_js
    let inp_idxs: ArrayD<usize> = ArrayD::from_shape_fn(self.input_dim.clone(), |i| flat_index(&self.input_dim, &i, N));
    let mut inp_arr = ArrayD::from_elem(self.input_dim.clone(), (0, 0));
    Zip::from(&mut inp_arr).and(&inp_idxs).for_each(|r, &a| {
      *r = (a, 0 as usize);
    });
    let mut ssig: Vec<Vec<Fr>> = inp_arr
      .map_axis(Axis(self.input_dim.ndim() - 1), |x| {
        construct_ssig(x.as_slice().unwrap(), N, last_inp_dim, &partitions, true)
      })
      .into_iter()
      .collect();

    let mut outp_arr = ArrayD::from_elem(self.output_dim.clone(), (0, 0));
    let outp_idxs: ArrayD<usize> = ArrayD::from_shape_fn(self.output_dim.clone(), |i| flat_index(&self.output_dim, &i, N) + offset);
    Zip::from(&mut outp_arr).and(&outp_idxs).and(&self.permutation).for_each(|r, &a, b| {
      *r = (a, flat_index(&self.input_dim, &b, N));
    });
    ssig.append(
      &mut outp_arr
        .map_axis(Axis(self.output_dim.ndim() - 1), |x| {
          construct_ssig(x.as_slice().unwrap(), N, last_outp_dim, &partitions, false)
        })
        .into_iter()
        .collect(),
    );
    let mut ssig_polys: Vec<_> = ssig.iter().map(|x| DensePolynomial::from_coefficients_vec(domain.ifft(x))).collect();

    let mut proof_0 = vec![L_i_x_1[0], sid_x];
    proof_0.append(&mut ssig_polys.iter().map(|x| util::msm::<G1Projective>(&srs.X1A, &x.coeffs)).collect());
    let mut proof_2 = vec![sid_poly];
    proof_2.append(&mut ssig_polys);

    return (proof_0, vec![L_i_x_2[0]], proof_2);
  }

  fn prove(
    &mut self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let input = inputs[0];
    let output = outputs[0];
    let input_len = input.first().unwrap().raw.len();
    let output_len = output.first().unwrap().raw.len();

    let N = max(input_len, output_len);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Round 1: quotients
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);

    // Calculate fjs
    let fj_polys: Vec<_> = inputs[0].iter().chain(outputs[0].iter()).map(|x| &x.poly).collect();
    let sid_poly = &setup.2[0];
    let ssig_polys = &setup.2[1..];
    let beta_poly = DensePolynomial::from_coefficients_vec(vec![beta]);
    let gamma_poly = DensePolynomial::from_coefficients_vec(vec![gamma]);

    let fj1_polys: Vec<_> = fj_polys
      .iter()
      .enumerate()
      .map(|(i, x)| {
        let offset = DensePolynomial::from_coefficients_vec(vec![beta * Fr::from((i * N) as i32)]);
        &sid_poly.mul(&beta_poly) + *x + offset + gamma_poly.clone()
      })
      .collect();

    // Calculate gjs
    let gj1_polys: Vec<_> = fj_polys.iter().enumerate().map(|(i, x)| &ssig_polys[i].mul(&beta_poly) + *x + gamma_poly.clone()).collect();

    let f1_poly = fj1_polys.iter().fold(DensePolynomial::from_coefficients_vec(vec![Fr::one()]), |acc, x| acc.mul(x));
    let g1_poly = gj1_polys.iter().fold(DensePolynomial::from_coefficients_vec(vec![Fr::one()]), |acc, x| acc.mul(x));

    // Calculate Z
    let mut Z = vec![Fr::zero(); N];
    Z[0] = Fr::one();
    for i in 1..N {
      let o = domain.element(i - 1);
      Z[i] = Z[i - 1] * f1_poly.evaluate(&o) * g1_poly.evaluate(&o).inverse().unwrap();
    }
    let Z_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&Z));
    let Z_x = util::msm::<G1Projective>(&srs.X1A, &Z_poly.coeffs);

    // Calculate quotient for L0(X)(Z(X)-1) = 0
    let mut L0_evals = vec![Fr::zero(); N];
    L0_evals[0] = Fr::one();
    let L0_poly = DensePolynomial {
      coeffs: domain.ifft(&L0_evals),
    };
    let one = DensePolynomial { coeffs: vec![Fr::one()] };
    let L0Z_poly = L0_poly.mul(&Z_poly.sub(&one));
    let L0Z_Q = L0Z_poly.divide_by_vanishing_poly(domain).unwrap();
    let L0Z_Q_x = util::msm::<G1Projective>(&srs.X1A, &L0Z_Q.0.coeffs);

    // Calculate quotient for Z(x)f'(x) = Z(gx)g'(x)
    let Zg_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    let lhs = f1_poly.mul(&Z_poly);
    let rhs = g1_poly.mul(&Zg_poly);
    let Q = lhs.sub(&rhs).divide_by_vanishing_poly(domain).unwrap();
    let Q_x = util::msm::<G1Projective>(&srs.X1A, &Q.0.coeffs);

    // Round 2: openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut rng2 = StdRng::from_entropy();
    let mut r: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    let proof = vec![Z_x, L0Z_Q_x, Q_x];
    let mut proof: Vec<_> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // Calculate opening argument for Z over omega * a
    let omega = domain.group_gen();
    let a = Fr::rand(rng);
    let Z_ga = Z_poly.evaluate(&(omega * a));
    let Z_ga_poly = DensePolynomial { coeffs: vec![Z_ga] };
    let Z_V = DensePolynomial {
      coeffs: vec![-a * omega, Fr::one()],
    };
    let temp = Z_poly.sub(&Z_ga_poly);
    let Z_Q: DensePolynomial<_> = &temp / &Z_V;
    let Z_Q_x = util::msm::<G1Projective>(&srs.X1A, &Z_Q.coeffs);

    // Calculate opening quotient for Z(x)f'(x) = Z(gx)g'(x) check
    let fj_poly_iter = fj_polys.iter().map(|x| *x);
    let mut q1_evals: Vec<Fr> = once(sid_poly).chain(ssig_polys.iter()).chain(fj_poly_iter.clone()).map(|p| p.evaluate(&a)).collect();

    let f1_a = f1_poly.evaluate(&a);
    let ssig_as = &q1_evals[1..ssig_polys.len() + 1];
    let fj_as = &q1_evals[ssig_polys.len() + 1..];
    let gj1_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| ssig_as[i] * beta + *x + gamma).collect();
    let a_pows = calc_pow(a, N);
    let g1_a: Fr = gj1_as[..gj1_as.len() - 1].iter().product();
    let f1_a_poly = DensePolynomial::from_coefficients_vec(vec![f1_a]);
    let lhs = Z_poly.mul(&f1_a_poly);
    let rhs_mul = DensePolynomial::from_coefficients_vec(vec![g1_a * Z_ga]);
    let rhs_add = DensePolynomial::from_coefficients_vec(vec![gamma + fj_as[fj_as.len() - 1]]);
    let rhs = (&ssig_polys[ssig_polys.len() - 1].mul(&beta_poly) + &rhs_add).mul(&rhs_mul);
    let v = DensePolynomial::from_coefficients_vec(vec![a_pows[N - 1] - Fr::one()]).mul(&Q.0);
    let q1_V = DensePolynomial { coeffs: vec![-a, Fr::one()] };
    let r_Q = &(&(&lhs - &rhs) - &v) / &q1_V;
    let r_Q_x = util::msm::<G1Projective>(&srs.X1A, &r_Q.coeffs);

    // Calculate opening argument for fjs, sid, ssigs over a
    let b = Fr::rand(rng);
    let bs = calc_pow(b, ssig_polys.len() + fj_polys.len());
    let q1_poly: DensePolynomial<Fr> = ssig_polys.iter().chain(fj_poly_iter).enumerate().fold(sid_poly.clone(), |acc, (i, p)| acc + p.mul(bs[i]));
    let q1_a = q1_poly.evaluate(&a);
    let q1_a_poly = DensePolynomial { coeffs: vec![q1_a] };
    let temp = q1_poly.sub(&q1_a_poly);
    let q1_Q = &temp / &q1_V;
    let q1_Q_x = util::msm::<G1Projective>(&srs.X1A, &q1_Q.coeffs);

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let mut r1: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    proof.append(&mut vec![q1_Q_x, Z_Q_x, r_Q_x].iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r1[i]).collect());
    r.append(&mut r1);
    let opening_r: Fr = inputs[0].iter().chain(outputs[0].iter()).enumerate().map(|(i, x)| x.r * bs[ssig_polys.len() + i]).sum();
    let mut C: Vec<G1Projective> = vec![
      setup.0[0] * r[0] - (srs.X1P[N] - srs.X1P[0]) * r[1],
      srs.X1P[0] * f1_a * r[0] - srs.X1P[0] * (a_pows[N - 1] - Fr::one()) * r[2] - (srs.X1P[1] - (srs.X1P[0] * a)) * r[5],
      srs.X1P[0] * opening_r - (srs.X1P[1] - (srs.X1P[0] * a)) * r[3],
      srs.X1P[0] * r[0] - (srs.X1P[1] - (srs.X1P[0] * a * omega)) * r[4],
    ];
    proof.append(&mut C);
    let mut s_1s = setup.0[1..].iter().map(|x| Into::<G1Projective>::into(*x)).collect();
    proof.append(&mut s_1s);

    let mut evals = vec![Z_ga];
    evals.append(&mut q1_evals);
    return (proof, vec![setup.1[0].into()], evals);
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: &mut ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = max(inputs[0].first().unwrap().len, outputs[0].first().unwrap().len);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let [Z_x, L0Z_Q_x, Q_x, q1_Q_x, Z_Q_x, r_Q_x, C1, C2, C3, C4, sid_x] = proof.0[..11] else {
      panic!("Wrong proof format")
    };
    let ssig_xs = &proof.0[11..];
    let L0 = proof.1[0];
    let Z_ga = proof.2[0];

    let q1_evals = &proof.2[1..];
    let sid_a = q1_evals[0];
    let ssig_as = &q1_evals[1..ssig_xs.len() + 1];
    let fj_as = &q1_evals[ssig_xs.len() + 1..];

    // Round 1 randomness
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![Z_x, L0Z_Q_x, Q_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let a = Fr::rand(rng);
    let b = Fr::rand(rng);

    // Check L0(x)(Z(x) - 1) = V(x)Q(x)
    checks.push(vec![
      ((Z_x - srs.X1A[0]).into(), L0),
      (-L0Z_Q_x, (srs.X2A[N] - srs.X2A[0]).into()),
      (-C1, srs.Y2A),
    ]);

    // Check Z(x)f'(x) = Z(gx)g'(x)
    let fj1_as: Vec<_> = fj_as
      .iter()
      .enumerate()
      .map(|(i, x)| {
        let offset = beta * Fr::from((i * N) as i32);
        sid_a * beta + *x + offset + gamma
      })
      .collect();

    let gj1_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| ssig_as[i] * beta + *x + gamma).collect();

    let f1_a: Fr = fj1_as.iter().product();
    let g1_a: Fr = gj1_as[..gj1_as.len() - 1].iter().product();
    let a_pows = calc_pow(a, N);
    let V_x: G2Affine = (srs.X2P[1] - srs.X2P[0] * a).into();
    checks.push(vec![
      (Z_x, (srs.X2A[0] * f1_a).into()),
      (-ssig_xs[ssig_xs.len() - 1], (srs.X2A[0] * g1_a * beta * Z_ga).into()),
      ((-srs.X1A[0] * g1_a * (fj_as[fj_as.len() - 1] + gamma) * Z_ga).into(), srs.X2A[0]),
      ((-Q_x * (a_pows[N - 1] - Fr::one())).into(), srs.X2A[0]),
      (-r_Q_x, V_x),
      (-C2, srs.Y2A),
    ]);

    // Check opening commitments over a
    let fj_xs: Vec<_> = inputs[0].iter().chain(outputs[0].iter()).map(|x| &x.g1).collect();
    let bs = calc_pow(b, ssig_xs.len() + fj_xs.len());
    let fj_xs_iter = fj_xs.iter().map(|x| *x);
    let q1_x: G1Projective = ssig_xs
      .iter()
      .chain(fj_xs_iter.clone())
      .enumerate()
      .fold(sid_x.into(), |acc, (i, x)| acc + Into::<G1Projective>::into(*x) * bs[i]);

    let q1_a: Fr = q1_evals[1..].iter().enumerate().fold(sid_a, |acc, (i, x)| acc + *x * bs[i]);

    let r_x: G1Affine = (srs.X1P[0] * q1_a).into();
    checks.push(vec![((q1_x - r_x).into(), srs.X2A[0]), (-q1_Q_x, V_x.into()), (-C3, srs.Y2A)]);

    // Check Z opening commitment
    let Z_ga_x: G1Affine = (srs.X1P[0] * Z_ga).into();
    let V_x: G2Affine = (srs.X2P[1] - srs.X2P[0] * omega * a).into();
    checks.push(vec![((Z_x - Z_ga_x).into(), srs.X2A[0]), (-Z_Q_x, V_x.into()), (-C4, srs.Y2A)]);
    checks
  }
}
