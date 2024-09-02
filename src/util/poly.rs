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

  (0..m).into_par_iter().map(|i| vecs[0][i] * vecs[1][i]).collect()
}

pub fn mul_two_polys(polys: &Vec<DensePolynomial<Fr>>) -> DensePolynomial<Fr> {
  assert!(polys.len() == 2);
  if polys[0].is_zero() || polys[1].is_zero() {
    return DensePolynomial::zero();
  }
  let N: usize = polys.iter().map(|p| p.coeffs.len()).sum();
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

  let p_evals = polys.par_iter().map(|p| domain.fft(&p.coeffs)).collect::<Vec<_>>();
  let p_evals = elementwise_product(&p_evals);
  DensePolynomial::from_coefficients_vec(domain.ifft(&p_evals))
}

// Multiply a list of polynomials in parallel
// TODO: explore if there exists a more efficient parallel algorithm
pub fn mul_polys(polys: &Vec<DensePolynomial<Fr>>) -> DensePolynomial<Fr> {
  // Base case: if the list has only one polynomial, return it directly
  if polys.len() == 1 {
    return polys[0].clone();
  }

  // Parallel recursive case: pairwise multiply the polynomials
  let next_level: Vec<DensePolynomial<Fr>> = polys
    .par_chunks(2) // Parallelize processing in chunks of 2
    .map(|chunk| {
      if chunk.len() == 2 {
        // If there are two polynomials in the chunk, multiply them
        mul_two_polys(&vec![chunk[0].clone(), chunk[1].clone()])
      } else {
        // If there's only one polynomial in the chunk, return it
        chunk[0].clone()
      }
    })
    .collect();

  // Recursively call mul_polys on the next level until we get the root
  mul_polys(&next_level)
}
