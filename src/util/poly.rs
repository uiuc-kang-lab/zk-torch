use ark_bn254::Fr;
use ark_ff::Zero;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::One;

pub fn mul_polys(polys: &Vec<DensePolynomial<Fr>>, domain_size: usize) -> DensePolynomial<Fr> {
  let domain = GeneralEvaluationDomain::new(domain_size * polys.len()).unwrap();
  if polys[0].is_zero() {
    return DensePolynomial::zero();
  }
  let mut p_evals = polys[0].evaluate_over_domain_by_ref(domain);
  for p in polys[1..].iter() {
    if p.is_zero() {
      return DensePolynomial::zero();
    } else {
      p_evals *= &p.evaluate_over_domain_by_ref(domain);
    }
  }
  p_evals.interpolate()
}
