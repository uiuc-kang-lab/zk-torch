use ark_ec::{VariableBaseMSM, pairing::Pairing};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain, Evaluations};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bn254};
use ark_std::{ops::Mul, ops::Sub};
use ndarray::{Array, IxDyn};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc,Tensor};
use crate::util;


// precompute freq_cis in llama
fn compute_cos_sin(N: usize, D: usize) -> (Tensor<Fr>, Tensor<Fr>) {
  let mut cos_vec = Vec::new();
  let mut sin_vec = Vec::new();
  for i in 0..N {
    for j in 0..D {
      let theta = i as f64 * (0.0001 as f64).powf(2. * (j as f64/2.).floor() as f64 / D as f64);
      let cos = theta.cos();
      let sin = theta.sin();
      cos_vec.push(Fr::from((cos*util::SCALE_FACTOR as f64).floor() as i64));
      sin_vec.push(Fr::from((sin*util::SCALE_FACTOR as f64).floor() as i64));
    }
  }
  let cos = Array::from_shape_vec(IxDyn(&[N, D]), cos_vec).unwrap();
  let sin = Array::from_shape_vec(IxDyn(&[N, D]), sin_vec).unwrap();
  (cos, sin)
}
  
// compute Q' or K' from Q or K
// FIXME: we need to prove this computation is correct in zk-llm, too.
fn compute_q_or_k_prime(input: &Tensor<Fr>) -> Tensor<Fr> {
  let input_shape = input.shape();
  let q_or_k = input.clone().into_raw_vec();
  let q_or_k = q_or_k.chunks(input_shape[1]).map(|x| x.to_vec()).collect::<Vec<_>>();
  // slice in odd indices
  let odd_indices: Vec<Vec<Fr>> = q_or_k.iter().cloned().map(|row| row.into_iter().skip(1).step_by(2).collect()).collect();
  
  // slice in even indices
  let even_indices: Vec<Vec<Fr>> = q_or_k.iter().cloned().map(|row| row.into_iter().step_by(2).collect()).collect();
  
  // stack them
  let result: Vec<Vec<Fr>> = odd_indices.iter().zip(even_indices.iter()).map(|(row_odd, row_even)| {
    row_odd.iter().zip(row_even.iter()).flat_map(|(&a, &b)| vec![a*Fr::from(-1), b]).collect()
  }).collect();
  
  let flat_result = result.into_iter().flat_map(|inner_vec| inner_vec).collect();
  let result = Array::from_shape_vec(IxDyn(&[input_shape[0], input_shape[1]]), flat_result).unwrap();
  result
}

pub struct RopeBasicBlock;
impl BasicBlock for RopeBasicBlock{
  type Proof = (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  type Setup = (Vec<G1Affine>,Vec<G2Affine>);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>> {
    
    // input size is N*D
    let N = inputs[0].shape()[0];
    let D = inputs[0].shape()[1];
    
    let inputs_prime = compute_q_or_k_prime(&inputs[0]);
    let (cos, sin) = compute_cos_sin(N, D);
    let mut rope = Vec::new();
    for i in 0..N {
      let mut result_row = Vec::new();
      for j in 0..D {
        result_row.push(cos[[i,j]]*inputs[0][[i, j]] + sin[[i,j]]*inputs_prime[[i,j]]);
      }
      rope.push(result_row);
    }
    let flat_rope = rope.into_iter().flat_map(|inner_vec| inner_vec).collect();
    vec![Array::from_shape_vec(IxDyn(&[N, D]), flat_rope).unwrap()]
  }

  fn setup(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           _model: &Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    // TODO: we can actually perform cos and sin pre-computation here
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   _setup: &Self::Setup,
                   _model: &Data,
                   inputs: &Vec<Data>,
                   output: &Data,
                   _rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    let N = inputs[0].raw.shape()[0];
    let D = inputs[0].raw.shape()[1];
    let domain  = GeneralEvaluationDomain::<Fr>::new(N*D).unwrap();
    // compute cos and sin
    let (cos, sin) = compute_cos_sin(N, D);
    let flat_cos = cos.clone().into_iter().collect::<Vec<_>>();
    let flat_sin = sin.clone().into_iter().collect::<Vec<_>>();
    let g_cos = Evaluations::from_vec_and_domain(flat_cos, domain).interpolate();
    let g_sin = Evaluations::from_vec_and_domain(flat_sin, domain).interpolate();
    let g_cos_x2 = G2Projective::msm(&srs.1[..N*D], &g_cos.coeffs).unwrap().into();
    let g_sin_x2 = G2Projective::msm(&srs.1[..N*D], &g_sin.coeffs).unwrap().into();
    // compute Q' from Q
    let q_or_k = inputs[0].raw.clone();
    let q_or_k_prime = compute_q_or_k_prime(&q_or_k);
    let input_prime_data = Data::new(srs,&q_or_k_prime);
    let t = (inputs[0].poly.mul(&g_cos) + input_prime_data.poly.mul(&g_sin)).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N*D-1], &t.coeffs).unwrap().into();
    return (vec![tx, input_prime_data.g1],vec![g_cos_x2, g_sin_x2],Vec::new());
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    output: &DataEnc,
                    proof: &Self::Proof,
                    _rng: &mut R){
    // Verify f(x)*g_cos(x)+f'(x)*g_sin(x)-h(x)=z(x)t(x)
    let lhs = Bn254::pairing(inputs[0].g1,proof.1[0]) + Bn254::pairing(proof.0[1],proof.1[1]) - Bn254::pairing(output.g1,srs.1[0]);
    let rhs = Bn254::pairing(proof.0[0],srs.1[inputs[0].len]-srs.1[0]);
    assert!(lhs==rhs);
  }
}

