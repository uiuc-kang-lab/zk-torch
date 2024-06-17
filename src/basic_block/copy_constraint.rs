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

fn flat_index(shape: &IxDyn, idx: &Option<IxDyn>, N: usize) -> Option<(usize, usize)> {
  assert!(*idx == None || shape.ndim() == idx.as_ref().unwrap().ndim());
  let mut product = vec![];
  // If inputs and outputs do not have the same last dimension, then the one
  // with the smaller dimension will have had their polynomials constructed from
  // a smaller evaluation domain. This indexing enables the smaller dimension's
  // roots of unity evaluation values to line up to the larger one.
  if let Some(j) = idx {
    let spacing = N / shape[shape.ndim() - 1];
    product.push(1);
    for d in 1..(shape.ndim() - 1) {
      product.push(product[d - 1] * shape[shape.ndim() - 1 - d]);
    }
    product = product.into_iter().rev().collect();
    let left_idx = if shape.ndim() == 1 {
      0
    } else {
      product.iter().enumerate().fold(0, |sum, (i, x)| *x * j[i] + sum)
    };
    let right_idx = j[shape.ndim() - 1] * spacing;
    Some((left_idx, right_idx))
  } else {
    None
  }
}

// Construct S_sigma_j polynomial evals over N-th roots of unity
// which encodes the permutation function
// In a copy-constraint, the permutation should form a cycle for all elements
// that should be the same over inputs and outputs.
// If is_input, idxs is [(flat_idx of the input index, 0)]. Otherwise,
// idxs is [(flat_idx of the output, flat_idx of the permuted input idx)]
fn construct_ssig(
  idxs: &[((usize, usize), Option<(usize, usize)>)],
  N: usize,
  last_dim: usize,
  partitions: &HashMap<Option<(usize, usize)>, Vec<(usize, usize)>>,
  is_input: bool,
) -> Vec<Fr> {
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  idxs
    .iter()
    .flat_map(|(idx, perm_idx)| {
      let inp_idx = if is_input { Some(*idx) } else { *perm_idx };
      let sigma = {
        let partition = partitions.get(&inp_idx).ok_or_else(|| format!("Key {:?} not found in the partition", inp_idx)).unwrap();
        if let Some(pos) = partition.iter().position(|x| *x == *idx) {
          Ok(partition[(pos + 1) % partition.len()])
        } else {
          Err(format!("Value {:?} not found in the list", *idx))
        }
      };
      let sigma = sigma.unwrap();
      let mut ssig = vec![Fr::from(sigma.0 as i32 + 1) * domain.element(sigma.1)];
      // Permute each filler element to itself
      let spacing = N / last_dim;
      let mut padding: Vec<_> = (1..spacing).map(|i| Fr::from(idx.0 as i32 + 1) * domain.element(idx.1 + i)).collect();
      ssig.append(&mut padding);
      ssig
    })
    .collect()
}

// Permutation has the same shape as the output, and each index stores the index of the input array it equals to
#[derive(Debug)]
pub struct CopyConstraintBasicBlock {
  pub permutation: ArrayD<Option<IxDyn>>,
  pub input_dim: IxDyn,
}

impl BasicBlock for CopyConstraintBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 1 && inputs[0].dim() == self.input_dim);
    vec![ArrayD::from_shape_fn(self.permutation.shape(), |i| {
      if let Some(idx) = &self.permutation[i] {
        inputs[0][idx]
      } else {
        Fr::zero()
      }
    })]
  }

  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let output_dim = self.permutation.dim();
    let last_inp_dim = self.input_dim[self.input_dim.ndim() - 1];
    let last_outp_dim = output_dim[output_dim.ndim() - 1];
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
    let offset: usize = input_dim.iter().take(self.input_dim.ndim() - 1).product();
    let flat_outp_idxs = self
      .permutation
      .indexed_iter()
      .map(|(i, _)| {
        let idx = flat_index(&output_dim, &Some(i), N).unwrap();
        (idx.0 + offset, idx.1)
      })
      .collect();
    let flat_outp_idxs = ArrayD::from_shape_vec(output_dim.clone(), flat_outp_idxs).unwrap();
    // Indices of the input which are in each position of the permutation
    let flat_perm_idxs = self.permutation.map(|i| flat_index(&self.input_dim, i, N));

    // Create partitions
    let mut partitions = HashMap::new();
    for i in indices(self.input_dim.clone()).into_iter() {
      let idx = flat_index(&self.input_dim, &Some(i), N);
      partitions.entry(idx).or_insert_with(|| Vec::from([idx.unwrap()]));
    }
    let mut pad = vec![];
    for (i, out_idx) in flat_outp_idxs.indexed_iter() {
      if let Some(perm_idx) = flat_perm_idxs[&i] {
        partitions.entry(Some(perm_idx)).or_insert_with(|| Vec::from([perm_idx])).push(*out_idx);
      } else {
        pad.push(*out_idx);
      }
    }
    if pad.len() > 0 {
      partitions.insert(None, pad);
    }

    // Calculate S_sigma_js
    let inp_idxs: ArrayD<(usize, usize)> = ArrayD::from_shape_fn(self.input_dim.clone(), |i| flat_index(&self.input_dim, &Some(i), N).unwrap());
    let mut inp_arr = ArrayD::from_elem(self.input_dim.clone(), ((0, 0), None));
    Zip::from(&mut inp_arr).and(&inp_idxs).for_each(|r, &a| {
      *r = (a, None);
    });
    let mut ssig: Vec<Vec<Fr>> = inp_arr
      .map_axis(Axis(self.input_dim.ndim() - 1), |x| {
        construct_ssig(x.as_slice().unwrap(), N, last_inp_dim, &partitions, true)
      })
      .into_iter()
      .collect();

    let mut outp_arr = ArrayD::from_elem(output_dim.clone(), ((0, 0), None));
    Zip::from(&mut outp_arr).and(&flat_outp_idxs).and(&self.permutation).for_each(|r, &a, b| {
      *r = (a, flat_index(&self.input_dim, b, N));
    });
    ssig.append(
      &mut outp_arr
        .map_axis(Axis(output_dim.ndim() - 1), |x| {
          construct_ssig(x.as_slice().unwrap(), N, last_outp_dim, &partitions, false)
        })
        .into_iter()
        .collect(),
    );
    let ssig_polys: Vec<_> = ssig.iter().map(|x| DensePolynomial::from_coefficients_vec(domain.ifft(x))).collect();

    // Get Lagrange basis from first None element
    let mut none_idx = 0;
    for i in indices(self.permutation.shape()) {
      let idx = i.clone();
      if self.permutation[i].is_none() {
        none_idx = N / last_outp_dim * idx[self.permutation.shape().len() - 1];
        break;
      }
    }

    let mut ssig_xs: Vec<_> = ssig_polys.iter().map(|x| util::msm::<G1Projective>(&srs.X1A, &x.coeffs)).collect();
    let mut proof_0 = vec![L_i_x_1[0], L_i_x_1[none_idx]];
    proof_0.append(&mut ssig_xs);

    return (proof_0, vec![L_i_x_2[0], L_i_x_2[none_idx]], ssig_polys);
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
    let input_deg = input.first().unwrap().raw.len();
    let output_deg = output.first().unwrap().raw.len();

    let N = max(input_deg, output_deg);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Round 1: quotients
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);

    // Calculate fjs
    let m = inputs[0].len() + outputs[0].len();
    let mut rng2 = StdRng::from_entropy();
    let fj_blind: Vec<_> = (0..m).map(|_| Fr::rand(&mut rng2)).collect();
    let fj_blind1: Vec<_> = (0..m).map(|_| Fr::rand(&mut rng2)).collect();
    let fj_blinds: Vec<_> = (0..m).map(|i| DensePolynomial::from_coefficients_vec(vec![fj_blind[i], fj_blind1[i]])).collect();
    let fj_polys: Vec<_> = inputs[0]
      .iter()
      .chain(outputs[0].iter())
      .enumerate()
      .map(|(i, x)| &x.poly + &fj_blinds[i].mul(&DensePolynomial::from(domain.vanishing_polynomial())))
      .collect();
    let mut fj_xs: Vec<_> = inputs[0]
      .iter()
      .chain(outputs[0].iter())
      .enumerate()
      .map(|(i, x)| (srs.X1P[N + 1] - srs.X1P[1]) * fj_blind1[i] + (srs.X1P[N] - srs.X1P[0]) * fj_blind[i] + x.g1)
      .collect();

    let ssig_polys = &setup.2[..];
    let beta_poly = DensePolynomial::from_coefficients_vec(vec![beta]);
    let gamma_poly = DensePolynomial::from_coefficients_vec(vec![gamma]);

    // Calculate Z
    let mut Z = vec![Fr::zero(); N];
    Z[0] = Fr::one();
    for j in 0..(N - 1) {
      let o = domain.element(j);
      let num: Fr = inputs[0]
        .iter()
        .chain(outputs[0].iter())
        .enumerate()
        .map(|(i, x)| beta * Fr::from((i + 1) as i32) * o + x.poly.evaluate(&o) + gamma)
        .product();
      let denom: Fr = inputs[0]
        .iter()
        .chain(outputs[0].iter())
        .enumerate()
        .map(|(i, x)| beta * ssig_polys[i].evaluate(&o) + x.poly.evaluate(&o) + gamma)
        .product();
      Z[j + 1] = Z[j] * num * denom.inverse().unwrap();
    }
    let Z_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&Z));
    let Z_blind: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    let Z_blind_poly = DensePolynomial::from_coefficients_vec(vec![Z_blind[0], Z_blind[1], Z_blind[2]]);
    let Z_poly = &Z_poly + &Z_blind_poly.mul(&DensePolynomial::from(domain.vanishing_polynomial()));
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

    // Calculate quotient for zero-pad check
    let outp_ndim = self.permutation.ndim();
    let mut has_none = false;
    let mut none_idx = IxDyn::zeros(outp_ndim); // position in f_polys

    let mut Lnone_f_Q_x = G1Projective::zero();
    for i in indices(self.permutation.shape()) {
      let idx = i.clone();
      if self.permutation[i].is_none() {
        has_none = true;
        none_idx = idx.clone();
        break;
      }
    }
    if has_none {
      let idx = flat_index(&self.permutation.dim(), &Some(none_idx.clone()), N).unwrap();
      let mut Lnone_evals = vec![Fr::zero(); N];
      Lnone_evals[idx.1] = Fr::one();
      let Lnone_poly = DensePolynomial {
        coeffs: domain.ifft(&Lnone_evals),
      };
      let fj_none_poly = &fj_polys[idx.0 + input.len()];
      let Lnone_f_poly = &Lnone_poly.mul(fj_none_poly);

      let Lnone_f_Q = Lnone_f_poly.divide_by_vanishing_poly(domain).unwrap();
      Lnone_f_Q_x = util::msm::<G1Projective>(&srs.X1A, &Lnone_f_Q.0.coeffs).into();
    }

    // Calculate quotient for Z(x)f'(x) = Z(gx)g'(x)
    let Zg_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    // These have the extra beta * X + gamma etc. terms that appear in Z, t, r
    let ft_polys: Vec<_> = fj_polys
      .iter()
      .enumerate()
      .map(|(i, x)| {
        let id_poly = DensePolynomial::from_coefficients_vec(vec![Fr::zero(), beta * Fr::from((i + 1) as i32)]);
        x + &id_poly + gamma_poly.clone()
      })
      .collect();
    let f1_poly = ft_polys.iter().fold(DensePolynomial::from_coefficients_vec(vec![Fr::one()]), |acc, x| acc.mul(x));
    let gt_polys: Vec<_> = fj_polys.iter().enumerate().map(|(i, x)| &ssig_polys[i].mul(&beta_poly) + x + gamma_poly.clone()).collect();
    let g1_poly = gt_polys.iter().fold(DensePolynomial::from_coefficients_vec(vec![Fr::one()]), |acc, x| acc.mul(x));
    let t_poly = f1_poly.mul(&Z_poly).sub(&g1_poly.mul(&Zg_poly));

    let t_poly = t_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_x = util::msm::<G1Projective>(&srs.X1A, &t_poly.coeffs);

    // Round 2: openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut rng2 = StdRng::from_entropy();
    let mut r: Vec<_> = (0..4).map(|_| Fr::rand(&mut rng2)).collect();
    let proof = vec![Z_x, L0Z_Q_x, t_x, Lnone_f_Q_x];
    let mut proof: Vec<_> = proof.iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r[i]).collect();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    let a = Fr::rand(rng);
    let mut fj_as: Vec<_> = fj_polys.iter().map(|p| p.evaluate(&a)).collect();

    // Calculate opening argument for Z over omega * a
    let omega = domain.group_gen();
    let Z_ga = Z_poly.evaluate(&(omega * a));
    let Z_ga_poly = DensePolynomial { coeffs: vec![Z_ga] };
    let Z_V = DensePolynomial {
      coeffs: vec![-a * omega, Fr::one()],
    };
    let temp = Z_poly.sub(&Z_ga_poly);
    let Z_Q: DensePolynomial<_> = &temp / &Z_V;
    let Z_Q_x = util::msm::<G1Projective>(&srs.X1A, &Z_Q.coeffs);

    // Calculate opening quotient for Z(x)f'(x) = Z(gx)g'(x) check
    let mut ssig_as: Vec<_> = ssig_polys.iter().map(|p| p.evaluate(&a)).collect();
    let ft_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| beta * Fr::from((i + 1) as i32) * a + *x + gamma).collect();
    let gt_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| ssig_as[i] * beta + *x + gamma).collect();
    let a_pows = calc_pow(a, N);
    let gt_a: Fr = gt_as[..gt_as.len() - 1].iter().product();
    let ft_a = ft_as.iter().product();
    let ft_a_poly = DensePolynomial::from_coefficients_vec(vec![ft_a]);
    let lhs = Z_poly.mul(&ft_a_poly);
    // assert!(util::msm::<G1Projective>(&srs.X1A, &lhs.coeffs) == Z_x * ft_a);
    let rhs_mul = DensePolynomial::from_coefficients_vec(vec![gt_a * Z_ga]);
    let rhs_add = DensePolynomial::from_coefficients_vec(vec![gamma + fj_as[fj_as.len() - 1]]);
    let rhs = (&ssig_polys[ssig_polys.len() - 1].mul(&beta_poly) + &rhs_add).mul(&rhs_mul);
    // let ssig_xs = &setup.0[2..];
    // assert!(
    //   util::msm::<G1Projective>(&srs.X1A, &rhs.coeffs)
    //     == (ssig_xs[ssig_xs.len() - 1] * beta + srs.X1A[0] * (fj_as[fj_as.len() - 1] + gamma)) * gt_a * Z_ga
    // );
    // assert!(DensePolynomial::from(domain.vanishing_polynomial()).evaluate(&a) == a_pows[N - 1] - Fr::one());
    let v = DensePolynomial::from_coefficients_vec(vec![a_pows[N - 1] - Fr::one()]).mul(&t_poly);
    // assert!(util::msm::<G1Projective>(&srs.X1A, &v.coeffs) == t_x * (a_pows[N - 1] - Fr::one()));
    let q1_V = DensePolynomial { coeffs: vec![-a, Fr::one()] };
    let r_poly = &(&lhs - &rhs) - &v;
    let r_Q = &r_poly / &q1_V;
    let r_Q_x = util::msm::<G1Projective>(&srs.X1A, &r_Q.coeffs);
    // assert!(r_Q.mul(&q1_V) == (&(&lhs - &rhs) - &v));
    // assert!(
    //   Bn254::pairing(util::msm::<G1Projective>(&srs.X1A, &(&(&lhs - &rhs) - &v).coeffs), srs.X2P[0])
    //     == Bn254::pairing(r_Q_x, srs.X2P[1] - srs.X2P[0] * a)
    // );

    // Calculate opening argument for fjs, ssigs over a
    let b = Fr::rand(rng);
    let bs = calc_pow(b, ssig_polys.len() + ft_polys.len());
    let q1_poly: DensePolynomial<Fr> =
      ssig_polys.iter().chain(fj_polys.iter()).enumerate().fold(DensePolynomial::zero(), |acc, (i, p)| acc + p.mul(bs[i]));
    let q1_a = q1_poly.evaluate(&a);
    let q1_a_poly = DensePolynomial { coeffs: vec![q1_a] };
    let temp = q1_poly.sub(&q1_a_poly);
    let q1_Q = &temp / &q1_V;
    let q1_Q_x = util::msm::<G1Projective>(&srs.X1A, &q1_Q.coeffs);

    // Blinding
    let mut r1: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    proof.append(&mut vec![q1_Q_x, Z_Q_x, r_Q_x].iter().enumerate().map(|(i, x)| (*x) + srs.Y1P * r1[i]).collect());
    r.append(&mut r1);
    let mut C: Vec<G1Projective> = vec![
      setup.0[0] * r[0] - (srs.X1P[N] - srs.X1P[0]) * r[1],
      -(srs.X1P[N] - srs.X1P[0]) * r[3],
      srs.X1P[0] * ft_a * r[0] - srs.X1P[0] * (a_pows[N - 1] - Fr::one()) * r[2] - (srs.X1P[1] - (srs.X1P[0] * a)) * r[6],
      -(srs.X1P[1] - (srs.X1P[0] * a)) * r[4],
      srs.X1P[0] * r[0] - (srs.X1P[1] - (srs.X1P[0] * a * omega)) * r[5],
    ];
    proof.append(&mut C);

    let mut ssig_xs = setup.0[2..].iter().map(|x| Into::<G1Projective>::into(*x)).collect();
    proof.append(&mut ssig_xs);
    proof.append(&mut fj_xs);

    let mut evals = vec![Z_ga];
    evals.append(&mut ssig_as);
    evals.append(&mut fj_as);
    return (proof, vec![setup.1[0].into(), setup.1[1].into()], evals);
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
    let input = inputs[0];

    let [Z_x, L0Z_Q_x, t_x, Lnone_f_Q_x, q1_Q_x, Z_Q_x, r_Q_x, C1, C2, C3, C4, C5] = proof.0[..12] else {
      panic!("Wrong proof format")
    };

    let m = inputs[0].len() + outputs[0].len();
    let ssig_xs = &proof.0[12..m + 12];
    let fj_xs = &proof.0[m + 12..];

    let [L0, Lnone] = proof.1[..] else { panic!("Wrong proof format") };
    let Z_ga = proof.2[0];

    let q1_evals = &proof.2[1..];
    let ssig_as = &q1_evals[..m];
    let fj_as = &q1_evals[m..];

    // Round 1 randomness
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![Z_x, L0Z_Q_x, t_x, Lnone_f_Q_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let a = Fr::rand(rng);
    let b = Fr::rand(rng);

    // Check L0(x)(Z(x) - 1) = V(x)Q(x)
    checks.push(vec![
      ((Z_x - srs.X1A[0]).into(), L0),
      (-L0Z_Q_x, (srs.X2A[N] - srs.X2A[0]).into()),
      (-C1, srs.Y2A),
    ]);

    // Check Lnone(x)f(x) = V(x)Q(x)
    for i in indices(self.permutation.shape()) {
      if self.permutation[i.clone()].is_none() {
        let idx = flat_index(&self.permutation.dim(), &Some(i), N).unwrap();
        let flat_none_f_idx = (idx.0 + input.len(), idx.1);
        checks.push(vec![
          (fj_xs[flat_none_f_idx.0], Lnone),
          (-Lnone_f_Q_x, (srs.X2A[N] - srs.X2A[0]).into()),
          (-C2, srs.Y2A),
        ]);
        break;
      }
    }

    // Check Z(x)f'(x) = Z(gx)g'(x)
    let ft_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| beta * Fr::from((i + 1) as i32) * a + *x + gamma).collect();
    let gt_as: Vec<_> = fj_as.iter().enumerate().map(|(i, x)| ssig_as[i] * beta + *x + gamma).collect();

    let ft_a: Fr = ft_as.iter().product();
    let gt_a: Fr = gt_as[..gt_as.len() - 1].iter().product();
    let a_pows = calc_pow(a, N);
    let V_x: G2Affine = (srs.X2P[1] - srs.X2P[0] * a).into();
    checks.push(vec![
      (Z_x, (srs.X2A[0] * ft_a).into()),
      (
        (-(ssig_xs[ssig_xs.len() - 1] * beta + srs.X1A[0] * (fj_as[fj_as.len() - 1] + gamma)) * gt_a * Z_ga).into(),
        srs.X2A[0],
      ),
      ((-t_x * (a_pows[N - 1] - Fr::one())).into(), srs.X2A[0]),
      (-r_Q_x, V_x),
      (-C3, srs.Y2A),
    ]);

    // Check opening commitments over a
    let bs = calc_pow(b, ssig_xs.len() + fj_xs.len());
    let q1_x: G1Projective = ssig_xs
      .iter()
      .chain(fj_xs.iter())
      .enumerate()
      .fold(G1Projective::zero(), |acc, (i, x)| acc + Into::<G1Projective>::into(*x) * bs[i]);

    let q1_a: Fr = q1_evals.iter().enumerate().fold(Fr::zero(), |acc, (i, x)| acc + *x * bs[i]);

    checks.push(vec![
      ((q1_x - srs.X1P[0] * q1_a).into(), srs.X2A[0]),
      (-q1_Q_x, V_x.into()),
      (-C4, srs.Y2A),
    ]);

    // Check Z opening commitment
    let Z_ga_x: G1Affine = (srs.X1P[0] * Z_ga).into();
    let V_x: G2Affine = (srs.X2P[1] - srs.X2P[0] * omega * a).into();
    checks.push(vec![((Z_x - Z_ga_x).into(), srs.X2A[0]), (-Z_Q_x, V_x.into()), (-C5, srs.Y2A)]);
    checks
  }
}
