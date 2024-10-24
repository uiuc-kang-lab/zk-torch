#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc, SRS};
use crate::{
  basic_block::{PairingCheck, ProveVerifyCache},
  util::{self, calc_pow},
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
  collections::{BTreeMap, HashMap},
  default,
  iter::{once, repeat, Map},
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
// idxs tuple (i, j) refers to the jth element in the ith polynomial based on
// the flattened input or output ArrayD
// If is_input, idxs is [(flat_idx of the input index, (0, 0))]. Otherwise,
// idxs is [(flat_idx of the output, flat_idx of the permuted input idx)]
fn construct_ssig(
  idxs: &[((usize, usize), Option<(usize, usize)>)],
  N: usize,
  last_dim: usize,
  sigma_map: &HashMap<Option<(usize, usize)>, HashMap<(usize, usize), (usize, usize)>>,
  is_input: bool,
) -> Vec<Fr> {
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  idxs
    .par_iter()
    .flat_map(|(idx, perm_idx)| {
      let inp_idx = if is_input { Some(*idx) } else { *perm_idx };
      let sigma = sigma_map.get(&inp_idx).unwrap().get(idx).unwrap();
      let mut ssig = vec![Fr::from(sigma.0 as i32 + 1) * domain.element(sigma.1)];
      // Permute each filler element to itself
      let spacing = N / last_dim;
      let mut padding: Vec<_> = (1..spacing).map(|i| Fr::from(idx.0 as i32 + 1) * domain.element(idx.1 + i)).collect();
      ssig.append(&mut padding);
      ssig
    })
    .collect()
}

// Returns the padding_partitions field for CopyConstraintBasicBlock when the given permutation padding elements are 0
fn zero_padding_partition(permutation: &ArrayD<Option<IxDyn>>) -> HashMap<Fr, Vec<IxDyn>> {
  let mut partition = vec![];
  for (i, _) in permutation.indexed_iter() {
    if permutation[&i] == None {
      partition.push(i);
    }
  }
  let mut padding_partition = HashMap::new();
  if partition.len() > 0 {
    padding_partition.insert(Fr::zero(), partition);
  }
  padding_partition
}

// Returns the padding partition where the non-zero padding value consists of all pad indices such that the last-axis subview containing it contains non-pad elements, and the zero padding value consists of all pad indices part of a last-axis subview containing only pad elements.
// If val is 0, then these will instead be combined.
fn max_padding_partitions(permutation: &ArrayD<Option<IxDyn>>, val: Fr) -> HashMap<Fr, Vec<IxDyn>> {
  let mut zero_indices = vec![];
  let mut nonzero_indices = vec![];
  for (i, subview) in permutation.axis_iter(Axis(permutation.ndim() - 1)).enumerate() {
    if subview.iter().all(|x| x.is_none()) {
      for (idx, _) in subview.indexed_iter() {
        let mut full_idx = idx.as_array_view().to_vec();
        full_idx.push(i);
        zero_indices.push(IxDyn(&full_idx));
      }
    } else {
      for (idx, val) in subview.indexed_iter() {
        if val.is_none() {
          let mut full_idx = idx.as_array_view().to_vec();
          full_idx.push(i);
          nonzero_indices.push(IxDyn(&full_idx));
        }
      }
    }
  }
  let mut partitions = HashMap::new();
  if val == Fr::zero() {
    zero_indices.append(&mut nonzero_indices);
  } else {
    if nonzero_indices.len() > 0 {
      partitions.insert(val, nonzero_indices);
    }
  }
  if zero_indices.len() > 0 {
    partitions.insert(Fr::zero(), zero_indices);
  }
  partitions
}

// Determines the scheme for padding partitions: pairs of (padding value, list of indices in the output containing that pad value)
#[derive(Debug, Default)]
pub enum PaddingEnum {
  #[default]
  Zero,
  Max(Fr),
}

// Gets padding partition based on the enum value
fn get_padding_partition(permutation: &ArrayD<Option<IxDyn>>, pad_type: &PaddingEnum) -> HashMap<Fr, Vec<IxDyn>> {
  match pad_type {
    PaddingEnum::Zero => zero_padding_partition(permutation),
    PaddingEnum::Max(val) => max_padding_partitions(permutation, *val),
  }
}

// This BasicBlock implements Plonk's copy constraint protocol over the inputs and outputs (Sec. 5.2 and 8 of https://eprint.iacr.org/2019/953.pdf) [1].
// permutation has the same shape as the output, and each index stores the index of the input array it equals to.
// To support padding, padding_partitions contains pairs of (padding value, list of indices in the output containing that pad value)
#[derive(Debug)]
pub struct CopyConstraintBasicBlock {
  pub permutation: ArrayD<Option<IxDyn>>,
  pub input_dim: IxDyn,
  pub padding_partition: PaddingEnum,
}

impl BasicBlock for CopyConstraintBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1 && inputs[0].dim() == self.input_dim);
    let padding_partitions = get_padding_partition(&self.permutation, &self.padding_partition);
    let tmp_hashmap: HashMap<IxDyn, Fr> = padding_partitions.iter().flat_map(|(k, v)| v.iter().map(|x| (x.clone(), *k))).collect();
    Ok(vec![ArrayD::from_shape_fn(self.permutation.shape(), |i| {
      if let Some(idx) = &self.permutation[&i] {
        inputs[0][idx]
      } else {
        tmp_hashmap[&i]
      }
    })])
  }

  #[cfg(not(feature = "mock_prove"))]
  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let output_dim = self.permutation.dim();
    let last_inp_dim = self.input_dim[self.input_dim.ndim() - 1];
    let last_outp_dim = output_dim[output_dim.ndim() - 1];
    let N = max(last_inp_dim, last_outp_dim);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

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

    // Create partitions (p. 22 of [1])
    let mut partitions = HashMap::new();
    for i in indices(self.input_dim.clone()).into_iter() {
      let idx = flat_index(&self.input_dim, &Some(i), N);
      partitions.entry(idx).or_insert_with(|| vec![Vec::from([idx.unwrap()])]);
    }
    for (i, out_idx) in flat_outp_idxs.indexed_iter() {
      if let Some(perm_idx) = flat_perm_idxs[&i] {
        let val = partitions.entry(Some(perm_idx)).or_insert_with(|| vec![Vec::from([perm_idx])]);
        val[0].push(*out_idx);
      }
    }
    // Add padding partitions to partition
    let mut pad_partitions = vec![];
    let padding_partitions = get_padding_partition(&self.permutation, &self.padding_partition);
    let mut padding_values = padding_partitions.keys().collect::<Vec<_>>();
    padding_values.sort();
    for v in padding_values.iter() {
      let p = padding_partitions.get(v).unwrap();
      let flat_idxs: Vec<_> = p.iter().map(|i| flat_outp_idxs[i]).collect();
      if flat_idxs.len() > 0 {
        pad_partitions.push(flat_idxs);
      }
    }
    partitions.insert(None, pad_partitions);

    // Calculate S_sigma_js (p. 27 of [1])
    let inp_idxs: ArrayD<(usize, usize)> = ArrayD::from_shape_fn(self.input_dim.clone(), |i| flat_index(&self.input_dim, &Some(i), N).unwrap());
    let mut inp_arr = ArrayD::from_elem(self.input_dim.clone(), ((0, 0), None));
    Zip::from(&mut inp_arr).and(&inp_idxs).for_each(|r, &a| {
      *r = (a, None);
    });
    // Construct sigma_map because this data structure is more efficient than directly iterating over partitions
    // Take the first copy constraint in ResNet18 as a concrete example: it took about 3 sec for the below code to run
    // while iterating over partitions directly took about 1187 sec.
    // For Some keys, the map value will contain a hashmap containing all idx tuples corresponding to the sigma value.
    // For None keys, the map value will contain a hashmap for each set of indices that should have the same sigma value.
    let sigma_map: HashMap<Option<(usize, usize)>, HashMap<(usize, usize), (usize, usize)>> = partitions
      .par_iter()
      .map(|(&inp_idx, partition)| {
        let v = partition
          .iter()
          .flat_map(move |cycle| {
            cycle.iter().enumerate().map(move |(pos, &x)| {
              let next_pos = (pos + 1) % cycle.len();
              (x, (cycle[next_pos].0, cycle[next_pos].1))
            })
          })
          .collect();
        (inp_idx, v)
      })
      .collect();
    let mut ssig: Vec<Vec<Fr>> = inp_arr
      .map_axis(Axis(self.input_dim.ndim() - 1), |x| {
        construct_ssig(x.as_slice().unwrap(), N, last_inp_dim, &sigma_map, true)
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
          construct_ssig(x.as_slice().unwrap(), N, last_outp_dim, &sigma_map, false)
        })
        .into_iter()
        .collect(),
    );
    let mut ssig_poly_evals: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(x.to_vec())).collect();
    let mut ssig_polys: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(domain.ifft(x))).collect();
    let ssig_xs: Vec<_> = ssig_polys.iter().map(|x| util::msm::<G1Projective>(&srs.X1A, &x.coeffs)).collect();
    ssig_polys.append(&mut ssig_poly_evals);

    return (ssig_xs, vec![], ssig_polys);
  }

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    let output_dim = self.permutation.dim().as_array_view().to_vec();
    let input_dim = self.input_dim.as_array_view().to_vec();
    let last_inp_dim = self.input_dim[self.input_dim.ndim() - 1];
    let last_outp_dim = output_dim[output_dim.len() - 1];

    let inp_dim_product: usize = input_dim[..input_dim.len() - 1].iter().product();
    let out_dim_product: usize = output_dim[..output_dim.len() - 1].iter().product();
    let N = max(last_inp_dim, last_outp_dim);
    let m = inp_dim_product + out_dim_product;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    let ssig: Vec<_> = (0..m).map(|i| (0..N).map(|j| Fr::from((i + 1) as i32) * domain.element(j)).collect::<Vec<_>>()).collect::<Vec<_>>();

    let mut ssig_poly_evals: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(x.to_vec())).collect();
    let mut ssig_polys: Vec<_> = ssig_poly_evals.clone();
    let ssig_xs: Vec<_> = ssig_polys.iter().map(|_x| srs.X1P[0].clone()).collect();
    ssig_polys.append(&mut ssig_poly_evals);

    return (ssig_xs, vec![], ssig_polys);
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let input = inputs[0];
    let output = outputs[0];
    let input_deg = input.first().unwrap().raw.len();
    let output_deg = output.first().unwrap().raw.len();

    let N = max(input_deg, output_deg);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Round 0: Evaluate over N-th roots of unity for I/O
    let mut io_evals: Vec<Vec<Fr>> = if N == input_deg {
      input.par_iter().map(|x| x.raw.clone()).collect()
    } else {
      input.par_iter().map(|x| domain.fft(&x.poly.coeffs)).collect()
    };
    let mut output_evals: Vec<Vec<Fr>> = if N == output_deg {
      output.par_iter().map(|x| x.raw.clone()).collect()
    } else {
      output.par_iter().map(|x| domain.fft(&x.poly.coeffs)).collect()
    };
    io_evals.append(&mut output_evals);

    // Round 1: Commit input and output polynomials
    // Calculate fjs (corresponds to fjs on p. 22 and a, b, c on p. 28 of [1])
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
    let ssig_polys_len = ssig_polys.len();
    let ssig_poly_evals = &ssig_polys[ssig_polys_len / 2..];
    let ssig_polys = &ssig_polys[..ssig_polys_len / 2];

    // Round 1.5: compute q(x)s = [fj_blind(x) - fj(x)] / (x^N - 1) for proving fj_blind(x) = fj(x)
    // Fiat Shamir
    let mut bytes = Vec::new();
    fj_xs.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    let w = Fr::rand(rng);
    let w_pows = util::calc_pow(w, m);
    let rs: Vec<_> = (0..m).map(|_| Fr::rand(&mut rng2)).collect();
    let qj_x: G1Projective = (0..m).map(|i| (srs.X1P[1] * -fj_blind1[i] - srs.X1P[0] * fj_blind[i] + srs.Y1P * rs[i]) * w_pows[i]).sum();
    let inp_outp_rs: Vec<_> = inputs[0].iter().chain(outputs[0].iter()).map(|x| x.r).collect();
    let r_plus_r_Q_x: G1Projective = (0..m).map(|i| ((srs.X1P[N] - srs.X1P[0]) * rs[i] - srs.X1P[0] * inp_outp_rs[i]) * w_pows[i]).sum();

    // Round 2: Commit Z (p. 28 of [1])
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof = vec![qj_x];
    proof.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);
    let beta_poly = DensePolynomial::from_coefficients_vec(vec![beta]);
    let gamma_poly = DensePolynomial::from_coefficients_vec(vec![gamma]);

    let mut Z = vec![Fr::zero(); N];
    Z[0] = Fr::one();
    for j in 0..(N - 1) {
      let o = domain.element(j);
      let num: Fr = (0..io_evals.len()).into_par_iter().map(|i| beta * Fr::from((i + 1) as i32) * o + io_evals[i][j] + gamma).product();
      let denom: Fr = (0..io_evals.len()).into_par_iter().map(|i| beta * ssig_poly_evals[i].coeffs[j] + io_evals[i][j] + gamma).product();
      Z[j + 1] = Z[j] * num * denom.inverse().unwrap();
    }
    let Z_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&Z));
    let Z_blind: Vec<_> = (0..3).map(|_| Fr::rand(&mut rng2)).collect();
    let Z_blind_poly = DensePolynomial::from_coefficients_vec(vec![Z_blind[0], Z_blind[1], Z_blind[2]]);
    let Z_poly = &Z_poly + &Z_blind_poly.mul(&DensePolynomial::from(domain.vanishing_polynomial()));
    let Z_x = util::msm::<G1Projective>(&srs.X1A, &Z_poly.coeffs);

    // Round 3: Commit t (batched quotient polynomial of the below polynomials, p. 29 of [1])
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut proof_1 = vec![Z_x];
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    proof.append(&mut proof_1);

    // Calculate L0(X)(Z(X)-1) polynomial
    let mut L0_evals = vec![Fr::zero(); N];
    L0_evals[0] = Fr::one();
    let L0_poly = DensePolynomial {
      coeffs: domain.ifft(&L0_evals),
    };
    let one = DensePolynomial { coeffs: vec![Fr::one()] };
    let L0Z_poly = L0_poly.mul(&Z_poly.sub(&one));

    // Calculate pad check polynomials
    // These polys check that the first idx of each padding partition contains the corresponding padding value. The rest of elements are enforced through the copy constraint polynomials
    let mut pad_vals = vec![];
    let mut fj_none_idxs = vec![]; // position in f_polys
    let mut Lnone_polys = vec![];
    let padding_partitions = get_padding_partition(&self.permutation, &self.padding_partition);
    let mut padding_values = padding_partitions.keys().cloned().collect::<Vec<_>>();
    padding_values.sort();
    for val in padding_values.iter() {
      let partition = &padding_partitions[val];
      let idx = &partition[0];
      let flat_none_idx = flat_index(&self.permutation.dim(), &Some(idx.clone()), N).unwrap();
      let mut Lnone = vec![Fr::zero(); N];
      Lnone[flat_none_idx.1] = Fr::one();
      pad_vals.push(val);
      Lnone_polys.push(DensePolynomial { coeffs: domain.ifft(&Lnone) });
      fj_none_idxs.push(flat_none_idx.0 + input.len());
    }

    // Calculate batched quotient for Z(x)f'(x) = Z(gx)g'(x) and above checks
    let alpha = Fr::rand(rng);
    let alpha_poly = DensePolynomial::from_coefficients_vec(vec![alpha]);
    // Compute Z(omega * X) polynomial
    let Zg_poly = DensePolynomial {
      coeffs: Z_poly.coeffs.par_iter().enumerate().map(|(i, x)| x * &domain.element(i)).collect(),
    };
    // These have the extra beta * X + gamma etc. terms that appear in Z, t, r (seen as terms of t on p. 29 of [1])
    let ft_polys: Vec<_> = fj_polys
      .par_iter()
      .enumerate()
      .map(|(i, x)| {
        let id_poly = DensePolynomial::from_coefficients_vec(vec![Fr::zero(), beta * Fr::from((i + 1) as i32)]);
        x + &id_poly + gamma_poly.clone()
      })
      .collect();
    let f1_poly = util::mul_polys(&ft_polys);
    let gt_polys: Vec<_> = fj_polys.par_iter().enumerate().map(|(i, x)| &ssig_polys[i].mul(&beta_poly) + x + gamma_poly.clone()).collect();
    let g1_poly = util::mul_polys(&gt_polys);

    let alpha_pows = calc_pow(alpha, Lnone_polys.len() + 1);
    let mut none_poly = DensePolynomial::<Fr>::zero();
    for i in 0..pad_vals.len() {
      let pow_alpha_poly = DensePolynomial::from_coefficients_vec(vec![alpha_pows[i + 1]]);
      let pad_poly = &fj_polys[fj_none_idxs[i]] - &DensePolynomial::from_coefficients_vec(vec![*pad_vals[i]]);
      none_poly = none_poly + Lnone_polys[i].mul(&pad_poly).mul(&pow_alpha_poly);
    }
    let t_poly = f1_poly.mul(&Z_poly).sub(&g1_poly.mul(&Zg_poly)) + L0Z_poly.mul(&alpha_poly) + none_poly;
    let t_poly = t_poly.divide_by_vanishing_poly(domain).unwrap().0;
    let t_polys = util::split_polynomial(&t_poly, srs.X1P.len());
    let t_xs: Vec<_> = t_polys.iter().map(|x| util::msm::<G1Projective>(&srs.X1A, &x.coeffs)).collect();

    // Round 4: Compute openings
    // Fiat-Shamir
    let mut bytes = Vec::new();
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..t_xs.len()).map(|_| Fr::rand(&mut rng2)).collect();
    let mut proof_1: Vec<_> = t_xs.iter().enumerate().map(|(i, x)| x + srs.Y1P * r[i]).collect();
    proof_1.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    proof.append(&mut proof_1);

    let zeta = Fr::rand(rng);
    let omega = domain.group_gen();
    let Z_gz = Z_poly.evaluate(&(omega * zeta));
    let L0_z = L0_poly.evaluate(&zeta);
    let Lnone_zs: Vec<_> = Lnone_polys.iter().map(|p| p.evaluate(&zeta)).collect();
    let ssig_zs: Vec<_> = ssig_polys.iter().map(|p| p.evaluate(&zeta)).collect();
    let fj_zs: Vec<_> = fj_polys.iter().map(|p| p.evaluate(&zeta)).collect();
    let mut evals = vec![Z_gz, L0_z];
    evals.append(&mut Lnone_zs.clone());
    evals.append(&mut ssig_zs.clone());
    evals.append(&mut fj_zs.clone());

    // Round 5: Commit opening proofs
    // Fiat-Shamir
    let mut bytes = Vec::new();
    evals.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);

    // Calculate opening argument for Z over omega * a (W_zetaomega on p. 30 of [1])
    let Z_gz_poly = DensePolynomial { coeffs: vec![Z_gz] };
    let Z_V = DensePolynomial {
      coeffs: vec![-zeta * omega, Fr::one()],
    };
    let temp = Z_poly.sub(&Z_gz_poly);
    let Z_Q: DensePolynomial<_> = &temp / &Z_V;
    let W_gx = util::msm::<G1Projective>(&srs.X1A, &Z_Q.coeffs);

    // Calculate opening quotient for batched quotient check (containing r, fjs, ssigs on p. 30 of [1])
    let ft_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| beta * Fr::from((i + 1) as i32) * zeta + *x + gamma).collect();
    let gt_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| ssig_zs[i] * beta + *x + gamma).collect();
    let zeta_pows = calc_pow(zeta, max(N, srs.X1P.len() * (t_xs.len() - 1)));
    let gt_z: Fr = gt_zs[..gt_zs.len() - 1].iter().product();
    let ft_z = ft_zs.iter().product();
    let ft_z_poly = DensePolynomial::from_coefficients_vec(vec![ft_z]);
    let lhs = Z_poly.mul(&ft_z_poly);
    let rhs_mul = DensePolynomial::from_coefficients_vec(vec![gt_z * Z_gz]);
    let rhs_add = DensePolynomial::from_coefficients_vec(vec![gamma + fj_zs[fj_zs.len() - 1]]);
    let rhs = (&ssig_polys[ssig_polys.len() - 1].mul(&beta_poly) + &rhs_add).mul(&rhs_mul);
    let q1_V = DensePolynomial {
      coeffs: vec![-zeta, Fr::one()],
    };

    // Compute linearization polynomial r (p. 30 of [1])
    let mut r_none_poly = DensePolynomial::<Fr>::zero();
    for i in 0..Lnone_zs.len() {
      let pad_val_poly = DensePolynomial::from_coefficients_vec(vec![*pad_vals[i]]);
      r_none_poly = &r_none_poly
        + &DensePolynomial::from_coefficients_vec(vec![Lnone_zs[i] * alpha_pows[i + 1]]).mul(&fj_polys[fj_none_idxs[i]].sub(&pad_val_poly));
    }
    let r_t_poly = &t_polys[0]
      + &t_polys[1..].iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, p)| {
        acc + p.mul(&DensePolynomial::from_coefficients_vec(vec![zeta_pows[(i + 1) * srs.X1P.len() - 1]]))
      });
    let r_poly = &(&lhs - &rhs) - &DensePolynomial::from_coefficients_vec(vec![zeta_pows[N - 1] - Fr::one()]).mul(&r_t_poly)
      + Z_poly.sub(&one).mul(&DensePolynomial::from_coefficients_vec(vec![alpha * L0_z]))
      + r_none_poly;

    // Calculate opening argument for W over a (W_zeta on p. 30 of [1])
    let v = Fr::rand(rng);
    let vs = calc_pow(v, ssig_polys.len() + ft_polys.len());
    let q1_poly: DensePolynomial<Fr> =
      ssig_polys.iter().chain(fj_polys.iter()).enumerate().fold(DensePolynomial::zero(), |acc, (i, p)| acc + p.mul(vs[i]));
    let q1_z = q1_poly.evaluate(&zeta);
    let q1_z_poly = DensePolynomial { coeffs: vec![q1_z] };
    let W_poly = &(q1_poly.sub(&q1_z_poly) + r_poly) / &q1_V;
    let W_x = util::msm::<G1Projective>(&srs.X1A, &W_poly.coeffs);

    // Round 5 end randomness
    let mut bytes = Vec::new();
    vec![W_x, W_gx].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _ = Fr::rand(rng);

    // Blinding
    proof.append(&mut vec![W_x, W_gx]);
    let mut t_b = r[0];
    for i in 1..r.len() {
      t_b += r[i] * zeta_pows[i * srs.X1P.len() - 1];
    }
    proof.push(srs.X1P[0] * (t_b * (zeta_pows[N - 1] - Fr::one())));
    proof.push(r_plus_r_Q_x);

    let mut ssig_xs = setup.0.iter().map(|x| Into::<G1Projective>::into(*x)).collect();
    proof.append(&mut ssig_xs);
    proof.append(&mut fj_xs);

    return (proof, vec![], evals);
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    let N = max(inputs[0].first().unwrap().len, outputs[0].first().unwrap().len);
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    // TODO: Currently the prover is insecure because it passes in
    // inputs and outputs to the verifier which do not withstand polynomial
    // interpolation attacks in the inputs and outputs arguments.
    // To fix this, we have to suppport a blinding scheme
    // that works with openings more generally and enable the Data and DataEnc
    // constructors to use the blinding scheme when appropriate.
    let input = inputs[0];

    let m = inputs[0].len() + outputs[0].len();
    let [qj_x, Z_x] = proof.0[..2] else { panic!("Wrong proof format") };
    let t_xs = &proof.0[2..proof.0.len() - 2 * m - 4];
    let W_x = proof.0[proof.0.len() - 2 * m - 4];
    let W_gx = proof.0[proof.0.len() - 2 * m - 3];
    let C1 = proof.0[proof.0.len() - 2 * m - 2];
    let C2 = proof.0[proof.0.len() - 2 * m - 1];

    let ssig_xs = &proof.0[proof.0.len() - 2 * m..proof.0.len() - m];
    let fj_xs = &proof.0[proof.0.len() - m..];

    // TODO: have verifier compute Lagrange basis evals
    let padding_partitions = get_padding_partition(&self.permutation, &self.padding_partition);
    let mut padding_values = padding_partitions.keys().cloned().collect::<Vec<_>>();
    padding_values.sort();
    let [Z_gz, L0_z] = proof.2[..2] else { panic!("Wrong proof format") };
    let none_len = padding_values.len();
    let Lnone_zs = &proof.2[2..2 + none_len];
    let q1_evals = &proof.2[2 + none_len..];
    let ssig_zs = &q1_evals[..m];
    let fj_zs = &q1_evals[m..];

    // Round 1.5 randomness
    let mut bytes = Vec::new();
    fj_xs.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let w = Fr::rand(rng);

    // Round 2 randomness
    let mut bytes = Vec::new();
    vec![qj_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);

    // Round 3 randomness
    let mut bytes = Vec::new();
    vec![Z_x].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let alpha = Fr::rand(rng);

    // Round 4 randomness
    let mut bytes = Vec::new();
    t_xs.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let zeta = Fr::rand(rng);

    // Round 5 randomness
    let mut bytes = Vec::new();
    proof.2.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let v = Fr::rand(rng);

    // Round 5 end randomness
    let mut bytes = Vec::new();
    vec![W_x, W_gx].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let u = Fr::rand(rng);

    // Get none index for Lnone(x)f(x) = V(x)Q(x) check
    let mut pad_vals = vec![];
    let mut fj_none_idxs = vec![];
    for val in padding_values.iter() {
      let partition = padding_partitions.get(val).unwrap();
      let idx = &partition[0];
      let flat_idx = flat_index(&self.permutation.dim(), &Some(idx.clone()), N).unwrap();
      fj_none_idxs.push(flat_idx.0 + input.len());
      pad_vals.push(val);
    }

    // Perform the batched check (p. 31 of [1])
    let ft_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| beta * Fr::from((i + 1) as i32) * zeta + *x + gamma).collect();
    let gt_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| ssig_zs[i] * beta + *x + gamma).collect();
    let ft_z: Fr = ft_zs.iter().product();
    // Contains all but the last
    let gt_z: Fr = gt_zs[..gt_zs.len() - 1].iter().product();
    let zeta_pows = calc_pow(zeta, max(N, srs.X1P.len() * (t_xs.len() - 1)));

    let alpha_pows = calc_pow(alpha, Lnone_zs.len() + 1);
    let mut t_x = t_xs[0];
    for i in 1..t_xs.len() {
      t_x = (t_x + t_xs[i] * zeta_pows[i * srs.X1P.len() - 1]).into();
    }
    let mut D = Z_x * (ft_z + alpha * L0_z + u) - ssig_xs[ssig_xs.len() - 1] * beta * gt_z * Z_gz - t_x * (zeta_pows[N - 1] - Fr::one());
    for i in 0..pad_vals.len() {
      let pad_x: G1Affine = (-srs.X1P[0] * pad_vals[i] + fj_xs[fj_none_idxs[i]]).into();
      D = D + pad_x * alpha_pows[i + 1] * Lnone_zs[i];
    }

    let vs = calc_pow(v, ssig_xs.len() + fj_xs.len());
    let q1_x: G1Projective = ssig_xs
      .iter()
      .chain(fj_xs.iter())
      .enumerate()
      .fold(G1Projective::zero(), |acc, (i, x)| acc + Into::<G1Projective>::into(*x) * vs[i]);
    let F = D + q1_x;
    let q1_z: Fr = q1_evals.iter().enumerate().fold(Fr::zero(), |acc, (i, x)| acc + *x * vs[i]);
    let r_0 = -L0_z * alpha - gt_z * (fj_zs[fj_zs.len() - 1] + gamma) * Z_gz;
    let E = srs.X1A[0] * (-r_0 + q1_z + u * Z_gz);

    let w_pows = util::calc_pow(w, m);
    let fj_sum: G1Projective = fj_xs.iter().enumerate().map(|(i, x)| *x * w_pows[i]).sum();
    let inp_outp_sum: G1Projective = inputs[0].iter().chain(outputs[0].iter()).enumerate().map(|(i, x)| x.g1 * w_pows[i]).sum();
    let check_fj_terms = inp_outp_sum - fj_sum;
    checks.push(vec![
      ((W_x + W_gx * u).into(), srs.X2A[1]),
      ((-(W_x * zeta + W_gx * u * omega * zeta + F - E) + check_fj_terms).into(), srs.X2A[0]),
      (-qj_x, (srs.X2A[N] - srs.X2A[0]).into()),
      ((-C1 + C2).into(), srs.Y2A),
    ]);
    checks
  }
}
