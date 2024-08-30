use ark_bn254::Fr;
use ark_ff::Zero;
use ark_poly::{DenseUVPolynomial, univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_std::One;
use rayon::prelude::*;

fn elementwise_product<T>(vecs: &[Vec<T>]) -> Vec<T>
where
    T: std::iter::Product + std::marker::Send + std::marker::Sync + Copy + std::ops::Mul<Output = T> + 'static,
{
    // Assuming vecs is non-empty and all vectors have the same length
    let m = vecs[0].len();

    (0..m).into_par_iter()
        .map(|i| {
            vecs.iter()
                .map(|v| v[i])
                .product()
        })
        .collect()
}

pub fn mul_polys(polys: &Vec<DensePolynomial<Fr>>) -> DensePolynomial<Fr> {
  let degree = polys.iter().map(|p| p.degree()).sum::<usize>();
  let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();
  if polys[0].is_zero() {
    return DensePolynomial::zero();
  }
  let p_evals = polys.par_iter().map(|poly| domain.fft(&poly.coeffs)).collect::<Vec<_>>();
  let p_eval = elementwise_product(&p_evals);
  
  DensePolynomial::from_coefficients_vec(domain.ifft(&p_eval))
}
