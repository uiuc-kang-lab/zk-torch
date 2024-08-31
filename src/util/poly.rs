use ark_bn254::Fr;
use ark_ff::Zero;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::One;
use rayon::prelude::*;

fn elementwise_product<T>(vecs: &[Vec<T>]) -> Vec<T>
where
  T: std::iter::Product + std::marker::Send + std::marker::Sync + Copy + std::ops::Mul<Output = T> + 'static,
{
  // Assuming vecs is non-empty and all vectors have the same length
  let m = vecs[0].len();

  (0..m).into_par_iter().map(|i| vecs.iter().map(|v| v[i]).product()).collect()
}

pub fn mul_polys(polys: &Vec<DensePolynomial<Fr>>) -> DensePolynomial<Fr> {
  let N: usize = polys.iter().map(|p| p.coeffs.len()).sum();
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  if polys[0].is_zero() {
    return DensePolynomial::zero();
  }
  let mut p_evals = domain.fft(&polys[0].coeffs);
  for p in polys[1..].iter() {
    if p.is_zero() {
      return DensePolynomial::zero();
    } else {
      p_evals = elementwise_product(&vec![p_evals, domain.fft(&p.coeffs)]);
    }
  }
  DensePolynomial::from_coefficients_vec(domain.ifft(&p_evals))
}
