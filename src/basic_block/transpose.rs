use ark_ec::{VariableBaseMSM, pairing::Pairing};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain, Evaluations};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bn254};
use ark_std::{ops::Mul, ops::Sub, One, Zero, UniformRand};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc,Tensor};

pub struct TransposeBasicBlock;
impl BasicBlock for TransposeBasicBlock{
  type Proof = (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  type Setup = (Vec<G1Affine>,Vec<G2Affine>);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>>{

    let input = inputs[0].clone();
    let input_transpose = input.t().to_owned();
    vec![input_transpose]
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
    let gamma = Fr::rand(rng);
    let N = inputs[0].raw.shape()[0];
    let D = inputs[0].raw.shape()[1];

    let mut z_tmp = Fr::one();
    let mut ab_tmp = Fr::one();
    let mut l1_array: Vec<Fr> = Vec::new();
    let mut z_array: Vec<Fr> = Vec::new();
    let mut z_array_: Vec<Fr> = Vec::new();
    let mut ab_array: Vec<Fr> = Vec::new();
    for n in 0..N {
      if n == 0{
        l1_array.push(Fr::one());
      } else {
        l1_array.push(Fr::zero());
      }
      z_array_.push(z_tmp);
      for d in 0..D{
        let a = inputs[0].raw[[n,d]]+beta*Fr::from((n*D+d) as i64)+gamma;
        //let b = output.raw[n*D+d]+beta*Fr::from((d*D+n) as i64)+gamma; // TODO: should be this one
        let b = inputs[0].raw[[d,n]]+beta*Fr::from((d*D+n) as i64)+gamma;
        let a_div_b = a/b;
        ab_tmp = ab_tmp*a_div_b;
      }
      z_tmp = z_tmp*ab_tmp;
      z_array.push(z_tmp);
      ab_array.push(ab_tmp);
      ab_tmp = Fr::one();
    }

    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();

    // TODO: need to check L_1(X)*(Z(X)-1)=0    
    // L_1(X)
    let l1 = Evaluations::from_vec_and_domain(l1_array, domain).interpolate();
    // a(X)/b(X)
    let f = Evaluations::from_vec_and_domain(ab_array, domain).interpolate();
    // Z(wX)
    let z = Evaluations::from_vec_and_domain(z_array, domain).interpolate();
    // Z(X)
    let z_ = Evaluations::from_vec_and_domain(z_array_, domain).interpolate();

    let l1x = G1Projective::msm_unchecked(&srs.0, &l1.coeffs).into();
    let fx = G1Projective::msm_unchecked(&srs.0, &f.coeffs).into(); 
    let gx = G1Projective::msm_unchecked(&srs.0, &z.coeffs).into();
    let gsx2 = G2Projective::msm(&srs.1[..N], &z_.coeffs).unwrap().into();
    
    // T(X) = [Z(wX)-Z(X)*(a(X)/b(X))]/[X^(N)-1]
    let t = z.sub(&f.mul(&z_)).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N-1], &t.coeffs).unwrap().into();
    return (vec![tx, gx, fx, l1x],vec![gsx2],Vec::new());
  }

  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    _output: &DataEnc,
                    proof: &Self::Proof,
                    _rng: &mut R){
    // Verify Z(wX)-Z(X)*(a(X)/b(X))=[X^(N)-1]T(X)
    let table_width = inputs[0].shape[0];
    let lhs = Bn254::pairing(proof.0[1],srs.1[0]) - Bn254::pairing(proof.0[2],proof.1[0]);
    let rhs = Bn254::pairing(proof.0[0],srs.1[table_width]-srs.1[0]);
    assert!(lhs==rhs);
    // Verify L_1(X)*(Z(X)-1)=0   
    // let l = Bn254::pairing(proof.0[3],proof.1[0]- srs.1[0]).is_zero();
    // assert!(l);
  }
}

