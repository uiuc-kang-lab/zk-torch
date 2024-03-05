use super::{BasicBlock, Data, DataEnc};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, VariableBaseMSM};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_std::{ops::Div, ops::Mul, ops::Sub, ops::Add, One, UniformRand, Zero};
use ndarray::ArrayD;
use rand::rngs::StdRng;

pub struct BridgeBasicBlock;
impl BasicBlock for BridgeBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> ArrayD<Fr> {
    inputs[0].clone()
  }
  fn setup(&self, srs: (&Vec<G1Affine>, &Vec<G2Affine>), model: &Data) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let N = model.raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let mut l0_array: Vec<Fr> = vec![Fr::zero(); N];
    let mut ln_minus_one_array: Vec<Fr> = vec![Fr::zero(); N];
    l0_array[0] = Fr::one();
    ln_minus_one_array[N - 1] = Fr::one();

    let l0 = Evaluations::from_vec_and_domain(l0_array, domain).interpolate();
    let ln_minus_one = Evaluations::from_vec_and_domain(ln_minus_one_array, domain).interpolate();

    let l0x2 = G2Projective::msm_unchecked(&srs.1[..N], &l0.coeffs).into();
    let lnx2 = G2Projective::msm_unchecked(&srs.1[..N], &ln_minus_one.coeffs).into();
    let omega_x2 = srs.1[0] * omega;
    (vec![], vec![l0x2, lnx2, omega_x2.into()])
  }
  fn prove(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    _output: &Data,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let beta2 = beta * beta;
    let N = inputs[0].raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let coeff = inputs[0].raw.clone().into_raw_vec();
    let mut coeff_f: DensePolynomial<Fr> = DenseUVPolynomial::from_coefficients_vec(coeff);
    let mut alpha_gen = Fr::one();
    for i in 1..inputs.len() {
      let coeff_f_i = DensePolynomial::from_coefficients_vec(inputs[i].raw.clone().into_raw_vec());
      alpha_gen *= alpha;
      coeff_f += &coeff_f_i.mul(alpha_gen);
    }
    let coeff_f = coeff_f;
    let coeff = &coeff_f.coeffs.to_vec();
    let open_value = coeff_f.evaluate(&omega);
    let open_value_poly = DensePolynomial::from_coefficients_vec(vec![open_value]);
    let coeff_com: G1Affine = G1Projective::msm_unchecked(&srs.0[..coeff.len()], coeff).into();
    let mut evals_com = inputs[0].g1;
    let mut alpha_gen = Fr::one();
    for i in 1..inputs.len() {
      alpha_gen *= alpha;
      evals_com = (inputs[i].g1 * alpha_gen + evals_com).into();
    }

    let mut x_vec = vec![Fr::zero(); N];
    x_vec[1] = Fr::one();
    let x = DenseUVPolynomial::from_coefficients_vec(x_vec);

    let mut l0_array: Vec<Fr> = vec![Fr::zero(); N];
    let mut ln_minus_one_array: Vec<Fr> = vec![Fr::zero(); N];
    l0_array[0] = Fr::one();
    ln_minus_one_array[N - 1] = Fr::one();

    let mut z_array: Vec<Fr> = Vec::new();
    let mut z_omega_array: Vec<Fr> = Vec::new();
    let mut z_tmp = Fr::zero();
    let mut b = Fr::one();
    for n in 0..N {
      z_array.push(z_tmp);
      let a = coeff_f.coeffs[n];
      z_tmp = z_tmp + a * b;
      b *= omega;
      z_omega_array.push(z_tmp);
    }
    assert!(z_omega_array[N - 1] - open_value == Fr::zero());

    // heck L_0(X)*Z(X)=0
    // L_0(X)
    let l0 = Evaluations::from_vec_and_domain(l0_array, domain).interpolate();
    let ln_minus_one = Evaluations::from_vec_and_domain(ln_minus_one_array, domain).interpolate();
    // Z(X)
    let z = Evaluations::from_vec_and_domain(z_array, domain).interpolate();
    // Z(wX)
    let z_omega = Evaluations::from_vec_and_domain(z_omega_array, domain).interpolate();
    //let mut z_omega = z.clone();
    //GeneralEvaluationDomain::<Fr>::distribute_powers(&mut z_omega.coeffs, omega);

    let zx = G1Projective::msm_unchecked(&srs.0[..N], &z.coeffs).into();
    let z_omegax = G1Projective::msm_unchecked(&srs.0[..N], &z_omega.coeffs).into();

    // T(X) = [Z(wX)-Z(X)-Xf(X)+beta*L_0(X)*Z(X)+beta2*L_n(X)*(Z(wX)-v)]/[X^(N)-1]
    let mut input_poly_sum = inputs[0].poly.clone();
    let mut alpha_gen = Fr::one();
    for i in 1..inputs.len() {
      alpha_gen *= alpha;
      let input_poly_i = inputs[i].poly.mul(alpha_gen);
      input_poly_sum = input_poly_sum.add(input_poly_i);
    }
    let t = (z_omega.sub(&z).sub(&input_poly_sum.mul(&x))
      + (&l0.mul(&z) * beta)
      + (&ln_minus_one.mul(&z_omega.sub(&open_value_poly)) * beta2))
      .divide_by_vanishing_poly(domain)
      .unwrap()
      .0;
    let tx = G1Projective::msm_unchecked(&srs.0[..N - 1], &t.coeffs).into();

    // Q(X) = [f'(X)-f'(w)]/ [X-w]
    let x_minus_omega = DenseUVPolynomial::from_coefficients_vec(vec![-omega, Fr::one()]);
    let mut f_prime_minus_f_prime_omega = coeff_f.clone();
    f_prime_minus_f_prime_omega.coeffs[0] -= open_value;
    let q = f_prime_minus_f_prime_omega.div(&x_minus_omega);
    let qx = G1Projective::msm_unchecked(&srs.0[..N - 1], &q.coeffs).into();

    let open_value_x = srs.0[0] * open_value;

    return (
      vec![tx, zx, z_omegax, coeff_com, evals_com.into(), qx, open_value_x.into()],
      setup.1.to_vec(),
    );
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    _output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let _alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let beta2 = beta * beta;
    // Verify [Z(wX)-Z(X)-Xf(X)+beta*L_0(X)*Z(X)+beta2*L_n(X)*(Z(wX)-v)]=[X^(N)-1]T(X)
    let N = inputs[0].shape[0];
    let lhs = Bn254::pairing(proof.0[2], srs.1[0]) - Bn254::pairing(proof.0[1], srs.1[0]) - Bn254::pairing(proof.0[4], srs.1[1])
      + Bn254::pairing(proof.0[1] * beta, proof.1[0])
      + Bn254::pairing(proof.0[2] * beta2 - proof.0[6] * beta2, proof.1[1]);
    let rhs = Bn254::pairing(proof.0[0], srs.1[N] - srs.1[0]);
    assert!(lhs == rhs);
    // Verify the opening f'(w) is valid
    let lhs = Bn254::pairing(proof.0[5], srs.1[1] * Fr::one() - proof.1[2]);
    let rhs = Bn254::pairing(proof.0[3] * Fr::one() - proof.0[6], srs.1[0]);
    assert!(lhs == rhs);
  }
}
