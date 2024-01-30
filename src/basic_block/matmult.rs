use ark_ec::VariableBaseMSM;
use ark_ff::Field;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::{MLSumcheck, Proof};
use ark_std::{UniformRand, rand::Rng, Zero, One};
use ark_std::rc::Rc;
use ndarray::{Array, IxDyn};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Affine};
use super::{BasicBlock,Data,DataEnc,Tensor};
use crate::util;

pub fn memoize<F: Field>(r: &Vec<F>, v: usize) -> Vec<F> {
	match v {
		1 => {
			vec![chi_step(0, r[v - 1]), chi_step(1, r[v - 1])]
		}
		_ => memoize(r, v - 1)
			.iter()
			.flat_map(|val| {
				[
					*val * &chi_step(0, r[v - 1]),
					*val * &chi_step(1, r[v - 1]),
				]
			})
			.collect(),
	}
}

pub fn chi_step<F: Field>(w: i128, x: F) -> F {
    match w {
        0 => F::one() - x,
        1 => x,
        _ => panic!("Invalid value for w"),
    }
}

pub fn dynamic_mle<F: Field>(fw: &Vec<F>, r: &Vec<F>, n: usize) -> Vec<F> {
	let chi_lookup = memoize(r, r.len());
    let mut final_result = Vec::new();
    for fw_chunk in fw.chunks(1<<n) {
        let result: F = fw_chunk
		.iter()
		.zip(chi_lookup.iter())
		.map(|(left, right)| *left * right)
		.sum();
        final_result.push(result);
    }
	
	final_result
}

pub struct MatMultBasicBlock;
impl BasicBlock for MatMultBasicBlock{
  type Setup = Result<(),()>;
  type Proof = (Proof<Fr>,PolynomialInfo,G1Affine,Fr,util::ZM_proof);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>> {
    let N = inputs[0].shape()[0];
    let matrix_a = &inputs[0];
    let matrix_b = &inputs[1];
    let mut r = Vec::new();
    for i in 0..N {
      for j in 0..N {
        let mut sum = Fr::zero();
        for k in 0..N {
          sum += matrix_a[[i,k]] * matrix_b[[k,j]];
        }
        r.push(sum);   
      }
    }
    vec![Array::from_shape_vec(IxDyn(&[matrix_a.shape()[0], matrix_b.shape()[1]]), r).unwrap()]
  }
  fn setup(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           _model: &Data) ->
           Result<(),()> {
    // MatMult do nothing in setup
    return Ok(());
  }
  fn prove<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   _setup: &Self::Setup,
                   _model: &Data,
                   inputs: &Vec<Data>,
                   output: &Data,
                   rng: &mut R) ->
                  Self::Proof{
    let N = inputs[0].raw.shape()[0];
    let n = (N as f64).log2() as usize;
    let mut point_0 = Vec::new();
    let mut point_1 = Vec::new();
    for _ in 0..n {
      point_0.push(Fr::rand(rng));
      point_1.push(Fr::rand(rng));
    }

    // Tried to accerlate the prover by using dynamic MLE, but need some more works
    //let f_b = DenseMultilinearExtension::from_evaluations_vec(n * 2, inputs[1].raw.clone());
    //let f_b = f_b.relabel(0, n, n);
    //let new_eval_b = dynamic_mle(&f_b.to_evaluations(), &point_1, n);
    //let f_b = DenseMultilinearExtension::from_evaluations_vec(n, new_eval_b);
    //let new_eval_a = dynamic_mle(&inputs[0].raw, &point_0, n);
    //let f_a = DenseMultilinearExtension::from_evaluations_vec(n, new_eval_a);

    let f_b = DenseMultilinearExtension::from_evaluations_vec(n * 2, inputs[1].raw.clone().into_raw_vec());
    let f_b = f_b.relabel(0, n, n);
    let f_b = f_b.fix_variables(&point_1);
    let f_a = DenseMultilinearExtension::from_evaluations_vec(n * 2, inputs[0].raw.clone().into_raw_vec());
    let f_a = f_a.fix_variables(&point_0);

    let f_c = DenseMultilinearExtension::from_evaluations_vec(n * 2, output.raw.clone().into_raw_vec());
    
    let point = point_0.iter().chain(point_1.iter()).cloned().collect::<Vec<_>>();
    let asserted_sum = f_c.evaluate(&point).unwrap();

    let uni_f_c = util::univariazation(&f_c);
    let uni_poly_coeff = &uni_f_c.coeffs;
    let com: G1Affine = G1Projective::msm(&srs.0[..uni_poly_coeff.len()], uni_poly_coeff).unwrap().into();

    let mut multiplicands = Vec::with_capacity(2);
    multiplicands.push(Rc::new(f_a));
    multiplicands.push(Rc::new(f_b));

    let mut poly = ListOfProductsOfPolynomials::new(n);
    poly.add_product(multiplicands.into_iter(), Fr::one());

    let poly_info = poly.info();
    let proof = MLSumcheck::prove(&poly).expect("fail to prove");
    let zm_proof = util::zm_prove(srs, &f_c, &point, asserted_sum, rng, util::NV_MAX);
    return (proof,poly_info,com,asserted_sum,zm_proof);
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    _output: &DataEnc,
                    proof: &Self::Proof,
                    rng: &mut R){
    let N = inputs[0].shape[0];
    let n = (N as f64).log2() as usize;
    let mut point_0 = Vec::new();
    let mut point_1 = Vec::new();
    for _ in 0..n {
      point_0.push(Fr::rand(rng));
      point_1.push(Fr::rand(rng));
    }
    let point = point_0.iter().chain(point_1.iter()).cloned().collect::<Vec<_>>();
    
    let com = proof.2;
    let asserted_sum = proof.3;
    let zm_proof = &proof.4;

    MLSumcheck::verify(&proof.1, asserted_sum, &proof.0).expect("fail to verify");
    util::zm_verify(srs, util::NV_MAX, point.len(), com, zm_proof, asserted_sum, &point, rng);
  }
}

