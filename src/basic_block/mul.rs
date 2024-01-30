use ark_ec::{VariableBaseMSM, pairing::Pairing};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bn254};
use ark_std::{ops::Mul, ops::Sub};
use ndarray::{Array, IxDyn};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc,Tensor};

pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock{
  type Proof = (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  type Setup = (Vec<G1Affine>,Vec<G2Affine>);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>> {
    let mut r = Vec::new();
    for i in 0..inputs[0].len(){
      r.push(inputs[0][i]*inputs[1][i]);
    }
    vec![Array::from_shape_vec(IxDyn(inputs[0].shape()), r).unwrap()]
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
                   output: &Data,
                   _rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    let N = inputs[0].raw.len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = G2Projective::msm(&srs.1[..N], &inputs[1].poly.coeffs).unwrap().into();
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N-1], &t.coeffs).unwrap().into();
    return (vec![tx],vec![gx2],Vec::new());
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    output: &DataEnc,
                    proof: &Self::Proof,
                    _rng: &mut R){
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bn254::pairing(inputs[0].g1,proof.1[0]) - Bn254::pairing(output.g1,srs.1[0]);
    let rhs = Bn254::pairing(proof.0[0],srs.1[inputs[0].len]-srs.1[0]);
    assert!(lhs==rhs);
    // Verify gx2
    let lhs = Bn254::pairing(inputs[1].g1,srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0],proof.1[0]);
    assert!(lhs==rhs);
  }
}

