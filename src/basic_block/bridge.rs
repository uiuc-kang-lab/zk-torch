use rand::Rng;
use ark_ec::{VariableBaseMSM, pairing::Pairing};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bn254};
use ark_std::{ops::Mul, ops::Sub, ops::Div, One, Zero, UniformRand};
use super::{BasicBlock,Data,DataEnc,Tensor};

pub struct BridgeBasicBlock;
impl BasicBlock for BridgeBasicBlock{
  type Setup = (Vec<G1Affine>,Vec<G2Affine>);
  type Proof = (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>> {
    inputs.to_vec()
  }
  fn setup(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           _model: &Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   _setup: &Self::Setup,
                   _model: &Data,
                   inputs: &Vec<Data>,
                   _output: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    let beta = Fr::rand(rng);
    let beta2 = beta*beta;
    let N = inputs[0].raw.len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let omega = domain.group_gen();

    let coeff = &inputs[0].raw.clone().into_raw_vec();
    let coeff_f: DensePolynomial<Fr> = DenseUVPolynomial::from_coefficients_vec(coeff.to_vec());
    let open_value = coeff_f.evaluate(&omega);
    let open_value_poly = DensePolynomial::from_coefficients_vec(vec![open_value]);
    let coeff_com: G1Affine = G1Projective::msm(&srs.0[..coeff.len()], coeff).unwrap().into();
    let evals_com = inputs[0].g1;
    
    let mut x_vec = vec![Fr::zero(); N];
    x_vec[1] = Fr::one();
    let x = DenseUVPolynomial::from_coefficients_vec(x_vec);

    let mut l0_array: Vec<Fr> = Vec::new();
    let mut ln_minus_one_array: Vec<Fr> = Vec::new();
    let mut z_array: Vec<Fr> = Vec::new();
    let mut z_omega_array: Vec<Fr> = Vec::new();
    let mut z_tmp = Fr::zero();
    let mut b = Fr::one();
    for n in 0..N {
      z_array.push(z_tmp);
      if n == 0{
        l0_array.push(Fr::one());
        ln_minus_one_array.push(Fr::zero());
      } else if n == N-1 {
        l0_array.push(Fr::zero());
        ln_minus_one_array.push(Fr::one());
      } else {
        l0_array.push(Fr::zero());
        ln_minus_one_array.push(Fr::zero());
      }
      let a = inputs[0].raw[n];
      z_tmp = z_tmp + a*b;
      b *= omega;
      z_omega_array.push(z_tmp);
    }
    assert!(z_omega_array[N-1]-open_value==Fr::zero());
    assert!(ln_minus_one_array[N-1]==Fr::one());
    assert!(ln_minus_one_array[N-2]==Fr::zero());

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
    
    let l0x2 = G2Projective::msm(&srs.1[..N], &l0.coeffs).unwrap().into();
    let lnx2 = G2Projective::msm(&srs.1[..N], &ln_minus_one.coeffs).unwrap().into();
    let zx = G1Projective::msm_unchecked(&srs.0[..N], &z.coeffs).into();
    let z_omegax = G1Projective::msm_unchecked(&srs.0[..N], &z_omega.coeffs).into();
    
    // T(X) = [Z(wX)-Z(X)-Xf(X)+beta*L_0(X)*Z(X)+beta2*L_n(X)*(Z(wX)-v)]/[X^(N)-1]
    let t = (z_omega.sub(&z).sub(&inputs[0].poly.mul(&x)) 
    + (&l0.mul(&z)*beta)
    + (&ln_minus_one.mul(&z_omega.sub(&open_value_poly))*beta2)).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm_unchecked(&srs.0[..N-1], &t.coeffs).into();

    // Q(X) = [f'(X)-f'(w)]/ [X-w]
    let x_minus_omega = DenseUVPolynomial::from_coefficients_vec(vec![-omega, Fr::one()]);
    let mut f_prime_minus_f_prime_omega = coeff_f.clone();
    f_prime_minus_f_prime_omega.coeffs[0] -= open_value;
    let q = f_prime_minus_f_prime_omega.div(&x_minus_omega);
    let qx = G1Projective::msm_unchecked(&srs.0[..N-1], &q.coeffs).into();

    return (vec![tx, zx, z_omegax, coeff_com, evals_com, qx],vec![l0x2, lnx2],vec![open_value, omega]);
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    _output: &DataEnc,
                    proof: &Self::Proof,
                    rng: &mut R){
    let beta = Fr::rand(rng);
    let beta2 = beta*beta;
    // Verify [Z(wX)-Z(X)-Xf(X)+beta*L_0(X)*Z(X)+beta2*L_n(X)*(Z(wX)-v)]=[X^(N)-1]T(X)
    let N = inputs[0].shape[0];
    let lhs = Bn254::pairing(proof.0[2],srs.1[0]) 
    - Bn254::pairing(proof.0[1],srs.1[0])
    - Bn254::pairing(proof.0[4],srs.1[1])
    + Bn254::pairing(proof.0[1] * beta,proof.1[0])
    + Bn254::pairing(proof.0[2] * beta2 - srs.0[0] * proof.2[0] * beta2,proof.1[1]);
    let rhs = Bn254::pairing(proof.0[0],srs.1[N]-srs.1[0]);
    assert!(lhs==rhs);
    // Verify the opening f'(w) is valid
    let lhs = Bn254::pairing(proof.0[5], srs.1[1]*Fr::one()-srs.1[0]*proof.2[1]);
    let rhs = Bn254::pairing(proof.0[3]*Fr::one()-srs.0[0]*proof.2[0], srs.1[0]);
    assert!(lhs==rhs);
  }
}

