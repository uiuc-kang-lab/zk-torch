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
use ndarray::{
  azip, indices, Array1, ArrayD, ArrayView, ArrayView1, ArrayView2, ArrayViewD, Axis, Dim, Dimension, IxDyn, IxDynImpl, NdIndex, Shape, Zip,
};
use rand::{rngs::StdRng, SeedableRng};
use rayon::{array, prelude::*};
use std::{
  cmp::{max, min},
  collections::{BTreeMap, HashMap},
  default,
  iter::{once, repeat, Map},
  mem,
};

fn swap_for_concat(ssig: &Vec<Vec<Fr>>, input_shapes: &Vec<Vec<usize>>, N_inp: Vec<usize>, N_out: usize) -> Vec<Vec<Fr>> {
  let mut ssig = ssig.clone();
  let input_dim_products = input_shapes
    .iter()
    .map(|shape| shape[..shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>())
    .collect::<Vec<_>>();
  let input_last_dims = input_shapes.iter().map(|shape| shape[shape.len() - 1]).collect::<Vec<_>>();
  let output_start_index = input_dim_products.iter().sum::<usize>();

  // i-th output poly
  for i in 0..input_dim_products[0] {
    let mut out_j = 0;
    for inp_id in 0..input_shapes.len() {
      for j in 0..input_last_dims[inp_id] {
        let step = N_out / N_inp[inp_id];
        let inp_j = j * step;
        let inp_i = input_dim_products[..inp_id].iter().sum::<usize>() + i;
        let out_i = output_start_index + i;
        swap_elements(&mut ssig, inp_i, inp_j, out_i, out_j);
        out_j += 1;
      }
    }
  }
  ssig
}

fn swap_elements<T>(table: &mut Vec<Vec<T>>, i1: usize, j1: usize, i2: usize, j2: usize) {
  if i1 == i2 && j1 == j2 {
    // Swapping the same element, no action needed
    return;
  }

  if i1 != i2 {
    if i1 > i2 {
      return swap_elements(table, i2, j2, i1, j1);
    }

    let (first_rows, rest) = table.split_at_mut(i2);
    let row1 = &mut first_rows[i1];
    let row2 = &mut rest[0];

    mem::swap(&mut row1[j1], &mut row2[j2]);
  } else {
    table[i1].swap(j1, j2);
  }
}

#[derive(Debug)]
pub struct ConcatLastDimBasicBlock {
  pub input_shapes: Vec<Vec<usize>>,
}

impl BasicBlock for ConcatLastDimBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    let axis = inputs[0].shape().len() - 1;
    let input_sliced = inputs.iter().enumerate().map(|(i, x)| util::slice_nd_array(x.to_owned().clone(), &self.input_shapes[i])).collect::<Vec<_>>();
    let r = ndarray::concatenate(Axis(axis), &input_sliced.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
    let r = util::pad_to_pow_of_two(&r, &Fr::zero());
    Ok(vec![r])
  }

  #[cfg(not(feature = "mock_prove"))]
  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    let mut output_shape = self.input_shapes[0].clone();
    let output_shape_len = output_shape.len();
    output_shape[output_shape_len - 1] = self.input_shapes.iter().map(|x| x[x.len() - 1]).sum();
    output_shape[output_shape_len - 1] = util::next_pow(output_shape[output_shape.len() - 1] as u32) as usize;

    let N_inps = self.input_shapes.iter().map(|shape| util::next_pow(shape[shape.len() - 1] as u32) as usize).collect::<Vec<_>>();
    let N_out = output_shape[output_shape.len() - 1];
    let domain = GeneralEvaluationDomain::<Fr>::new(N_out).unwrap();

    // m is the number of polynomials
    let output_dim_product = output_shape[..output_shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>();
    let input_dim_products = self
      .input_shapes
      .iter()
      .map(|shape| shape[..shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>())
      .collect::<Vec<_>>();
    let m = output_dim_product + input_dim_products.iter().sum::<usize>();

    let ssig: Vec<_> = (0..m).map(|i| (0..N_out).map(|j| Fr::from((i + 1) as i32) * domain.element(j)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let ssig = swap_for_concat(&ssig, &self.input_shapes, N_inps, N_out);
    let mut ssig_poly_evals: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(x.to_vec())).collect();
    let mut ssig_polys: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(domain.ifft(x))).collect();
    let ssig_xs: Vec<_> = ssig_polys.iter().map(|x| util::msm::<G1Projective>(&srs.X1A, &x.coeffs)).collect();
    ssig_polys.append(&mut ssig_poly_evals);

    return (ssig_xs, vec![], ssig_polys);
  }

  #[cfg(feature = "mock_prove")]
  fn setup(&self, srs: &SRS, _model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>) {
    eprintln!("\x1b[93mWARNING\x1b[0m: MockSetup is enabled. This is only for testing purposes.");
    let mut output_shape = self.input_shapes[0].clone();
    let output_shape_len = output_shape.len();
    output_shape[output_shape_len - 1] = self.input_shapes.iter().map(|x| x[x.len() - 1]).sum();
    output_shape[output_shape_len - 1] = util::next_pow(output_shape[output_shape.len() - 1] as u32) as usize;

    let N_out = output_shape[output_shape.len() - 1];
    let domain = GeneralEvaluationDomain::<Fr>::new(N_out).unwrap();

    // m is the number of polynomials
    let output_dim_product = output_shape[..output_shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>();
    let input_dim_products = self
      .input_shapes
      .iter()
      .map(|shape| shape[..shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>())
      .collect::<Vec<_>>();
    let m = output_dim_product + input_dim_products.iter().sum::<usize>();

    let ssig: Vec<_> = (0..m).map(|i| (0..N_out).map(|j| Fr::from((i + 1) as i32) * domain.element(j)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mut ssig_poly_evals: Vec<_> = ssig.par_iter().map(|x| DensePolynomial::from_coefficients_vec(x.to_vec())).collect();
    let mut ssig_polys: Vec<_> = ssig_poly_evals.clone();
    let ssig_xs: Vec<_> = (0..m).map(|_| srs.X1P[0].clone()).collect();
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
    assert!(outputs.len() == 1);
    let N = outputs[0].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // Round 0: Evaluate over N-th roots of unity for I/O
    let mut arrays = inputs.clone();
    arrays.push(outputs[0]);

    let mut io_evals: Vec<Vec<Vec<Fr>>> = inputs.iter().map(|input| input.par_iter().map(|x| domain.fft(&x.poly.coeffs)).collect()).collect();
    let output_evals: Vec<Vec<Fr>> = outputs[0].par_iter().map(|x| x.raw.clone()).collect();
    io_evals.push(output_evals);
    let io_evals: Vec<Vec<_>> = io_evals.into_iter().flatten().collect();
    println!("io_evals len: {}", io_evals.len());

    // m is the number of polynomials
    let m = io_evals.len();

    // Round 1: Commit inputs and output polynomials
    // Calculate fjs (corresponds to fjs on p. 22 and a, b, c on p. 28 of [1])
    let mut rng2 = StdRng::from_entropy();
    let fj_blind: Vec<_> = (0..m).map(|_| Fr::rand(&mut rng2)).collect();
    let fj_blind1: Vec<_> = (0..m).map(|_| Fr::rand(&mut rng2)).collect();
    let fj_blinds: Vec<_> = (0..m).map(|i| DensePolynomial::from_coefficients_vec(vec![fj_blind[i], fj_blind1[i]])).collect();
    let mut idx = 0;
    let fj_polys: Vec<DensePolynomial<Fr>> = arrays
      .iter()
      .map(|array| {
        array
          .iter()
          .map(|x| {
            idx += 1;
            &x.poly + &fj_blinds[idx - 1].mul(&DensePolynomial::from(domain.vanishing_polynomial()))
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
      .collect();
    let mut idx = 0;
    let mut fj_xs: Vec<_> = arrays
      .iter()
      .map(|array| {
        array
          .iter()
          .map(|x| {
            idx += 1;
            (srs.X1P[N + 1] - srs.X1P[1]) * fj_blind1[idx - 1] + (srs.X1P[N] - srs.X1P[0]) * fj_blind[idx - 1] + x.g1
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
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
    let inp_outp_rs: Vec<_> =
      arrays.iter().map(|array| array.iter().map(|x| x.r).collect::<Vec<_>>()).collect::<Vec<_>>().into_iter().flatten().collect();
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

    // TODO: Calculate pad check polynomials

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

    let t_poly = f1_poly.mul(&Z_poly).sub(&g1_poly.mul(&Zg_poly)) + L0Z_poly.mul(&alpha_poly);
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
    let ssig_zs: Vec<_> = ssig_polys.iter().map(|p| p.evaluate(&zeta)).collect();
    let fj_zs: Vec<_> = fj_polys.iter().map(|p| p.evaluate(&zeta)).collect();
    let mut evals = vec![Z_gz, L0_z];
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
    let gt_z: Fr = gt_zs.iter().product();
    let ft_z = ft_zs.iter().product();
    let ft_z_poly = DensePolynomial::from_coefficients_vec(vec![ft_z]);
    let lhs = Z_poly.mul(&ft_z_poly);
    let rhs = DensePolynomial::from_coefficients_vec(vec![gt_z * Z_gz]);
    let q1_V = DensePolynomial {
      coeffs: vec![-zeta, Fr::one()],
    };

    // Compute linearization polynomial r (p. 30 of [1])
    let r_t_poly = &t_polys[0]
      + &t_polys[1..].iter().enumerate().fold(DensePolynomial::zero(), |acc, (i, p)| {
        acc + p.mul(&DensePolynomial::from_coefficients_vec(vec![zeta_pows[(i + 1) * srs.X1P.len() - 1]]))
      });
    let r_poly = &(&lhs - &rhs) - &DensePolynomial::from_coefficients_vec(vec![zeta_pows[N - 1] - Fr::one()]).mul(&r_t_poly)
      + Z_poly.sub(&one).mul(&DensePolynomial::from_coefficients_vec(vec![alpha * L0_z]));

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
    let N = outputs[0].first().unwrap().len;
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    // TODO: Currently the prover is insecure because it passes in
    // inputs and outputs to the verifier which do not withstand polynomial
    // interpolation attacks in the inputs and outputs arguments.
    // To fix this, we have to suppport a blinding scheme
    // that works with openings more generally and enable the Data and DataEnc
    // constructors to use the blinding scheme when appropriate.

    let mut output_shape = self.input_shapes[0].clone();
    let output_shape_len = output_shape.len();
    output_shape[output_shape_len - 1] = self.input_shapes.iter().map(|x| x[x.len() - 1]).sum();
    output_shape[output_shape_len - 1] = util::next_pow(output_shape[output_shape.len() - 1] as u32) as usize;
    let output_dim_product = output_shape[..output_shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>();
    let input_dim_products = self
      .input_shapes
      .iter()
      .map(|shape| shape[..shape.len() - 1].iter().map(|v| util::next_pow(*v as u32) as usize).product::<usize>())
      .collect::<Vec<_>>();
    let m = output_dim_product + input_dim_products.iter().sum::<usize>();

    let [qj_x, Z_x] = proof.0[..2] else { panic!("Wrong proof format") };
    let t_xs = &proof.0[2..proof.0.len() - 2 * m - 4];
    let W_x = proof.0[proof.0.len() - 2 * m - 4];
    let W_gx = proof.0[proof.0.len() - 2 * m - 3];
    let C1 = proof.0[proof.0.len() - 2 * m - 2];
    let C2 = proof.0[proof.0.len() - 2 * m - 1];

    let ssig_xs = &proof.0[proof.0.len() - 2 * m..proof.0.len() - m];
    let fj_xs = &proof.0[proof.0.len() - m..];

    // TODO: have verifier compute Lagrange basis evals
    let [Z_gz, L0_z] = proof.2[..2] else { panic!("Wrong proof format") };
    let q1_evals = &proof.2[2..];
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

    // Perform the batched check (p. 31 of [1])
    let ft_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| beta * Fr::from((i + 1) as i32) * zeta + *x + gamma).collect();
    let gt_zs: Vec<_> = fj_zs.iter().enumerate().map(|(i, x)| ssig_zs[i] * beta + *x + gamma).collect();
    let ft_z: Fr = ft_zs.iter().product();
    let gt_z: Fr = gt_zs.iter().product();
    let zeta_pows = calc_pow(zeta, max(N, srs.X1P.len() * (t_xs.len() - 1)));

    // Compute linearization polynomial r (p. 30 of [1])
    let mut t_x = t_xs[0];
    for i in 1..t_xs.len() {
      t_x = (t_x + t_xs[i] * zeta_pows[i * srs.X1P.len() - 1]).into();
    }
    let D = Z_x * (ft_z + alpha * L0_z + u) - t_x * (zeta_pows[N - 1] - Fr::one());

    let vs = calc_pow(v, ssig_xs.len() + fj_xs.len());
    let q1_x: G1Projective = ssig_xs
      .iter()
      .chain(fj_xs.iter())
      .enumerate()
      .fold(G1Projective::zero(), |acc, (i, x)| acc + Into::<G1Projective>::into(*x) * vs[i]);
    let F = D + q1_x;
    let q1_z: Fr = q1_evals.iter().enumerate().fold(Fr::zero(), |acc, (i, x)| acc + *x * vs[i]);
    let r_0 = -L0_z * alpha - gt_z * Z_gz;
    let E = srs.X1A[0] * (-r_0 + q1_z + u * Z_gz);

    let w_pows = util::calc_pow(w, m);
    let fj_sum: G1Projective = fj_xs.iter().enumerate().map(|(i, x)| *x * w_pows[i]).sum();

    let mut m_counter = 0;
    let mut inp_outp_sum = G1Projective::zero();
    for input in inputs {
      for x in input.iter() {
        inp_outp_sum = inp_outp_sum + x.g1 * w_pows[m_counter];
        m_counter += 1;
      }
    }
    for x in outputs[0].iter() {
      inp_outp_sum = inp_outp_sum + x.g1 * w_pows[m_counter];
      m_counter += 1;
    }

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

// support concat over any dim except for the last
#[derive(Debug)]
pub struct ConcatBasicBlock {
  pub axis: usize,
}

impl BasicBlock for ConcatBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(self.axis != inputs[0].shape().len() - 1);
    let r = ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
    Ok(vec![r])
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, _outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    if inputs[0].ndim() == 0 {
      let r_vec = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<Data>>();
      let r = Array1::from_vec(r_vec).into_dyn();
      vec![r]
    } else {
      let r = ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
      vec![r]
    }
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    if inputs[0].ndim() == 0 {
      let r = inputs.iter().map(|input| input.first().unwrap().clone()).collect::<Vec<DataEnc>>();
      let r_enc = outputs[0];
      for i in 0..r.len() {
        assert!(r[i] == r_enc[i]);
      }
    } else {
      assert!(ndarray::concatenate(Axis(self.axis), &inputs.iter().map(|x| x.view()).collect::<Vec<_>>()) == Ok(outputs[0].clone()));
    }
    vec![]
  }
}
