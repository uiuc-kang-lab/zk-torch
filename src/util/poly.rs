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
  let domain = GeneralEvaluationDomain::<Fr>::new(N);
  if domain.is_some() {
    let domain = domain.unwrap();
    let p_evals = polys.par_iter().map(|p| domain.fft(&p.coeffs)).collect::<Vec<_>>();
    let p_evals = elementwise_product(&p_evals);
    DensePolynomial::from_coefficients_vec(domain.ifft(&p_evals))
  } else {
    karatsuba_multiply(&polys[0], &polys[1])
  }
}

fn mul_by_xn(poly: &DensePolynomial<Fr>, n: usize) -> DensePolynomial<Fr> {
  let mut new_coeffs = vec![Fr::zero(); n];
  new_coeffs.extend(poly.coeffs().iter().cloned());
  DensePolynomial::from_coefficients_vec(new_coeffs)
}

fn karatsuba_multiply(a: &DensePolynomial<Fr>, b: &DensePolynomial<Fr>) -> DensePolynomial<Fr> {
  let n = std::cmp::max(a.degree(), b.degree()) + 1;
  if n <= 1 << 27 {
    return a * b;
  }
  let m = n / 2;
  let (a0, a1) = karatsuba_split(a, m);
  let (b0, b1) = karatsuba_split(b, m);
  let z0 = karatsuba_multiply(&a0, &b0);
  let z2 = karatsuba_multiply(&a1, &b1);
  let a0_plus_a1 = &a0 + &a1;
  let b0_plus_b1 = &b0 + &b1;
  let z1 = karatsuba_multiply(&a0_plus_a1, &b0_plus_b1);
  let mut result = mul_by_xn(&z2, 2 * m);
  result += &z0;
  result = &result + &mul_by_xn(&(&(&z1 - &z2) - &z0), m);
  result
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

fn karatsuba_split(p: &DensePolynomial<Fr>, m: usize) -> (DensePolynomial<Fr>, DensePolynomial<Fr>) {
  let coeffs = p.coeffs();
  let low = DensePolynomial::from_coefficients_vec(coeffs[..m.min(coeffs.len())].to_vec());
  let high = DensePolynomial::from_coefficients_vec(coeffs[m.min(coeffs.len())..].to_vec());
  (low, high)
}

// Split poly into degree n-1 polynomials
pub fn split_polynomial(poly: &DensePolynomial<Fr>, n: usize) -> Vec<DensePolynomial<Fr>> {
  poly.coeffs().chunks(n).map(|chunk| DensePolynomial::from_coefficients_vec(chunk.to_vec())).collect()
}
