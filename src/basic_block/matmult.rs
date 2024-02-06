use ark_ec::VariableBaseMSM;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::{MLSumcheck, Proof};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_std::{UniformRand, rand::Rng, Zero, One};
use ark_std::rc::Rc;
use rand::{rngs::StdRng,SeedableRng};
use ndarray::{Array, IxDyn};
use ark_bn254::{Fr, G1Projective, G1Affine, G2Affine};
use super::{BasicBlock,Data,DataEnc,Tensor};
use crate::util;

pub struct MatMultBasicBlock;
impl BasicBlock for MatMultBasicBlock{
  type Setup = Result<(),()>;
  type Proof = (Proof<Fr>,PolynomialInfo,Fr,Fr,util::ZM_proof,util::ZM_proof,util::ZM_proof,util::ZM_proof,G1Affine,G1Affine,G1Affine,G1Affine,Fr);
  fn run(_model: &Vec<Tensor<Fr>>,
         inputs: &Vec<Tensor<Fr>>) ->
        Vec<Tensor<Fr>> {
    // Matrix shapes: MxN * NxK = MxK
    let matrix_a = &inputs[0];
    let matrix_b = &inputs[1];
    let (matrix_a, matrix_b) = (util::mat_padding(matrix_a), util::mat_padding(matrix_b));
    let M = matrix_a.shape()[0];
    let N = matrix_a.shape()[1];
    let K = matrix_b.shape()[1];
    let mut r = Vec::new();
    // matrix multiplication
    for i in 0..M {
      for j in 0..K {
        let mut sum = Fr::zero();
        for k in 0..N {
          sum += matrix_a[[i, k]] * matrix_b[[k, j]];
        }
        r.push(sum);
      }
    }
    vec![Array::from_shape_vec(IxDyn(&[M, K]), r).unwrap()]
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
    // Matrix shapes: MxN * NxK = MxK
    let matrix_a = &inputs[0].raw;
    let matrix_b = &inputs[1].raw;
    let (matrix_a, matrix_b) = (util::mat_padding(matrix_a), util::mat_padding(matrix_b));
    let matrix_c = &output.raw;
    let M = matrix_a.shape()[0];
    let N = matrix_a.shape()[1];
    let K = matrix_b.shape()[1];
    let m = (M as f64).log2() as usize;
    let n = (N as f64).log2() as usize;
    let k = (K as f64).log2() as usize;

    // Verifier's random points
    let mut point = Vec::new();
    for _ in 0..(m+k) {
      point.push(Fr::rand(rng));
    }
    let tho = Fr::rand(rng);

    // DenseMultilinearExtension is in little endian form so a matrix is stored in column major order (i.e., f(column, row))
    let f_b = DenseMultilinearExtension::from_evaluations_vec(n + k, matrix_b.clone().into_raw_vec());
    let f_b_fix = f_b.clone().fix_variables(&point[0..k]);
    let f_a = DenseMultilinearExtension::from_evaluations_vec(m + n, matrix_a.clone().into_raw_vec());
    let f_a_fix = util::fix_last_variables(&f_a, &point[k..m+k]);
    let f_c = DenseMultilinearExtension::from_evaluations_vec(m + k, matrix_c.clone().into_raw_vec());
    let f_c_value = f_c.evaluate(&point).unwrap();
    let f_c_zm_proof = util::zm_prove(srs, &f_c, &point, f_c_value, rng, util::NV_MAX);
    let mut rng1 = StdRng::from_entropy();
    let (random_poly_rc, random_poly, random_poly_sum) = util::random_polynomial(n, &mut rng1);

    let uni_f_c = util::univariazation(&f_c);
    let uni_f_c_coeff = &uni_f_c.coeffs;
    let com_f_c: G1Affine = G1Projective::msm(&srs.0[..uni_f_c_coeff.len()], uni_f_c_coeff).unwrap().into();
    let uni_f_b = util::univariazation(&f_b);
    let uni_f_b_coeff = &uni_f_b.coeffs;
    let com_f_b: G1Affine = G1Projective::msm(&srs.0[..uni_f_b_coeff.len()], uni_f_b_coeff).unwrap().into();
    let uni_f_a = util::univariazation(&f_a);
    let uni_f_a_coeff = &uni_f_a.coeffs;
    let com_f_a: G1Affine = G1Projective::msm(&srs.0[..uni_f_a_coeff.len()], uni_f_a_coeff).unwrap().into();
    let uni_random_poly = util::univariazation(&random_poly[0]);
    let uni_random_poly_coeff = &uni_random_poly.coeffs;
    let com_random_poly: G1Affine = G1Projective::msm(&srs.0[..uni_random_poly_coeff.len()], uni_random_poly_coeff).unwrap().into();

    let mut multiplicands = Vec::with_capacity(2);
    multiplicands.push(Rc::new(f_a_fix));
    multiplicands.push(Rc::new(f_b_fix));

    
    let mut poly = ListOfProductsOfPolynomials::new(n);
    poly.add_product(multiplicands.into_iter(), Fr::one());
    poly.add_product(random_poly_rc.into_iter(), tho);

    let asserted_sum = f_c_value + random_poly_sum * tho;

    let mut fs_rng = Blake2s512Rng::setup();
    let poly_info = poly.info();
    let (sumcheck_proof, _prover_state) =
        MLSumcheck::prove_as_subprotocol(&mut fs_rng, &poly).expect("fail to prove");

    // In zkVPD, the prover will evaluate the polynomial at the random point for verifier since verifier has no ability to evaluate the polynomial
    let mut fs_rng = Blake2s512Rng::setup();
    let subclaim = MLSumcheck::verify_as_subprotocol(&mut fs_rng, &poly_info, asserted_sum, &sumcheck_proof).expect("fail to verify");
    let verifier_random_point = subclaim.point;
    // concat verifier's random point with prover's random point
    let point_fb = point[..k].to_vec().iter().chain(verifier_random_point.iter()).cloned().collect::<Vec<_>>();
    let point_fa = verifier_random_point.iter().chain(point[k..m+k].iter()).cloned().collect::<Vec<_>>();
    let f_b_value = f_b.evaluate(&point_fb).unwrap();
    let f_b_zm_proof = util::zm_prove(srs, &f_b, &point_fb, f_b_value, rng, util::NV_MAX);
    let f_a_value = f_a.evaluate(&point_fa).unwrap();
    let f_a_zm_proof = util::zm_prove(srs, &f_a, &point_fa, f_a_value, rng, util::NV_MAX);
    let random_poly_value = random_poly[0].evaluate(&verifier_random_point).unwrap();
    let random_poly_zm_proof = util::zm_prove(srs, &random_poly[0], &verifier_random_point, random_poly_value, rng, util::NV_MAX);
    
    return (sumcheck_proof,poly_info,asserted_sum,random_poly_sum, f_a_zm_proof, f_b_zm_proof, f_c_zm_proof, random_poly_zm_proof, com_f_a, com_f_b, com_f_c, com_random_poly, subclaim.expected_evaluation);
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    _output: &DataEnc,
                    proof: &Self::Proof,
                    rng: &mut R){
    let M = inputs[0].shape[0];
    let K = inputs[1].shape[1];
    let m = (M as f64).log2() as usize;
    let k = (K as f64).log2() as usize;

    let mut point = Vec::new();
    for _ in 0..(m+k) {
      point.push(Fr::rand(rng));
    }
    let tho = Fr::rand(rng);
    let sumcheck_proof = &proof.0;
    let poly_info = &proof.1;
    let mut asserted_sum = proof.2;
    let random_poly_sum = proof.3;
    let f_a_zm_proof = &proof.4;
    let f_b_zm_proof = &proof.5;
    let f_c_zm_proof = &proof.6;
    let random_poly_zm_proof = &proof.7;
    let com_f_a = proof.8;
    let com_f_b = proof.9;
    let com_f_c = proof.10;
    let com_random_poly = proof.11;
    let expected_evaluation = proof.12;

    // Verifier can verify the sumcheck proof by herself
    let mut fs_rng = Blake2s512Rng::setup();
    MLSumcheck::verify_as_subprotocol(&mut fs_rng, &poly_info, asserted_sum, &sumcheck_proof).expect("fail to verify");
    
    // check four openings are correct
    assert!(f_a_zm_proof.value*f_b_zm_proof.value+tho*random_poly_zm_proof.value == expected_evaluation, "wrong subclaim");
    asserted_sum -= tho * random_poly_sum;
    util::zm_verify(srs, util::NV_MAX, point.len(), com_f_c, f_c_zm_proof, rng);
    util::zm_verify(srs, util::NV_MAX, f_b_zm_proof.point.len(), com_f_b, f_b_zm_proof, rng);
    util::zm_verify(srs, util::NV_MAX, f_a_zm_proof.point.len(), com_f_a, f_a_zm_proof, rng);
    util::zm_verify(srs, util::NV_MAX, random_poly_zm_proof.point.len(), com_random_poly, random_poly_zm_proof, rng);
  }
}

