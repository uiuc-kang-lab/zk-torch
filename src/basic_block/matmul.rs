#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use super::{BasicBlock, Data, DataEnc};
use crate::util;
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use rand::{rngs::StdRng, SeedableRng};

// Inputs to basic block are v,r_0,r_1,... where r_0,r_1,... are the rows of a matrix M
// Output of basic block is w = vM
// Proof steps:
// 1. M is converted to the vector "flat" where flat = alpha^0 r_0 + alpha^1 r_1 + ...
// 2. flat is pointwise multiplied by v to create the vector A
// 3. w is pointwise multiplied by the vector pow = [alpha^0, alpha^1, ...] to create the vector B
// 4. ∑A=∑B is checked by via A(0) and B(0)
struct AProof {
  x: G1Affine,        // A(x)
  Q_x: G1Affine,      // flat(x) * v(x) - A(x) = Q(x)Z(x)
  zero: G1Affine,     // A(0)
  zero_div: G1Affine, // (A(x)-A(0))/x
}
struct BProof {
  x: G1Affine,        // B(x)
  Q_x: G1Affine,      // w(x) * pow(x) - B(x) = Q(x)Z(x)
  zero_div: G1Affine, // (B(x)-B(0))/x
}

pub struct MatMulBasicBlock;
impl BasicBlock for MatMulBasicBlock {
  fn run(&self, _model: &Vec<Fr>, inputs: &Vec<&Vec<Fr>>) -> Vec<Fr> {
    let m = inputs.len() - 1;
    let n = inputs[0].len();
    let mut r = vec![Fr::zero(); m];
    for i in 0..m {
      for j in 0..n {
        r[i] += inputs[1 + i][j] * inputs[0][j];
      }
    }
    return r;
  }
  fn prove(
    &mut self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<&Data>,
    output: &Data,
    rng: &mut StdRng,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    let m = inputs.len() - 1;
    let n = inputs[0].raw.len();
    let domain_n = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let alpha = Fr::rand(rng);

    // Calculate pow
    let mut pow: Vec<Fr> = vec![Fr::one(); m];
    for i in 0..m - 1 {
      pow[i + 1] = pow[i] * alpha;
    }
    let pow_poly = DensePolynomial { coeffs: domain_m.ifft(&pow) };
    let pow_x = util::msm::<G1Projective>(&srs.0, &pow_poly.coeffs);

    // Calculate flat
    let mut flat = vec![Fr::zero(); n];
    let mut flat_r = Fr::zero();
    for i in 0..m {
      for j in 0..n {
        flat[j] += inputs[1 + i].raw[j] * pow[i];
      }
      flat_r += inputs[1 + i].r * pow[i];
    }
    let flat_poly = DensePolynomial {
      coeffs: domain_n.ifft(&flat),
    };
    let flat_x = util::msm::<G1Projective>(&srs.0, &flat_poly.coeffs);

    // Calculate A
    let A_i: Vec<Fr> = (0..n).map(|i| flat[i] * inputs[0].raw[i]).collect();
    let A_poly = DensePolynomial { coeffs: domain_n.ifft(&A_i) };
    let A_Q_poly = flat_poly.mul(&inputs[0].poly).sub(&A_poly).divide_by_vanishing_poly(domain_n).unwrap().0;
    let A = AProof {
      x: util::msm::<G1Projective>(&srs.0, &A_poly.coeffs).into(),
      Q_x: util::msm::<G1Projective>(&srs.0, &A_Q_poly.coeffs).into(),
      zero: (srs.0[0] * (Fr::from(n as u32).inverse().unwrap() * A_i.iter().sum::<Fr>())).into(),
      zero_div: util::msm::<G1Projective>(&srs.0, &A_poly.coeffs[1..]).into(),
    };
    let v_x_2 = util::msm::<G2Projective>(&srs.1, &inputs[0].poly.coeffs) + srs.1[srs.1.len() - 1] * inputs[0].r;
    let v_x_2 = v_x_2.into();

    // Calculate B
    let B_i: Vec<Fr> = (0..m).map(|i| output.raw[i] * pow[i]).collect();
    let B_poly = DensePolynomial { coeffs: domain_m.ifft(&B_i) };
    let B_Q_poly = output.poly.mul(&pow_poly).sub(&B_poly).divide_by_vanishing_poly(domain_m).unwrap().0;
    let B = BProof {
      x: util::msm::<G1Projective>(&srs.0, &B_poly.coeffs).into(),
      Q_x: util::msm::<G1Projective>(&srs.0, &B_Q_poly.coeffs).into(),
      zero_div: util::msm::<G1Projective>(&srs.0, &B_poly.coeffs[1..]).into(),
    };

    // Blinding
    let mut rng2 = StdRng::from_entropy();
    let r: Vec<_> = (0..7).map(|_| Fr::rand(&mut rng2)).collect();
    let proof: Vec<G1Affine> = vec![A.x, A.Q_x, A.zero, A.zero_div, B.x, B.Q_x, B.zero_div];
    let mut proof: Vec<G1Affine> = proof.iter().enumerate().map(|(i, x)| ((*x) + srs.0[srs.1.len() - 1] * r[i]).into()).collect();
    let C = vec![
      -(srs.0[n] - srs.0[0]) * r[1] + inputs[0].g1 * flat_r + flat_x * inputs[0].r + srs.0[srs.1.len() - 1] * inputs[0].r * flat_r - srs.0[0] * r[0],
      -srs.0[1] * r[3] + srs.0[0] * (r[0] - r[2]),
      -(srs.0[m] - srs.0[0]) * r[5] + pow_x * output.r - srs.0[0] * r[4],
      -srs.0[1] * r[6] + srs.0[0] * (r[4] - r[2] * Fr::from(n as u32) * Fr::from(m as u32).inverse().unwrap()),
    ];
    let mut C: Vec<G1Affine> = C.iter().map(|x| (*x).into()).collect();
    proof.append(&mut C);

    return (proof, vec![v_x_2]);
  }
  fn verify(
    &self,
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    model: &DataEnc,
    inputs: &Vec<&DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    rng: &mut StdRng,
  ) {
    let m = inputs.len() - 1;
    let n = inputs[0].len;
    let domain_m = GeneralEvaluationDomain::<Fr>::new(m).unwrap();
    let A = AProof {
      x: proof.0[0],
      Q_x: proof.0[1],
      zero: proof.0[2],
      zero_div: proof.0[3],
    };
    let B = BProof {
      x: proof.0[4],
      Q_x: proof.0[5],
      zero_div: proof.0[6],
    };
    let [C1, C2, C3, C4] = proof.0[7..] else { panic!("Wrong proof format") };
    let v_x_2 = proof.1[0];

    let alpha = Fr::rand(rng);

    // Calculate pow
    let mut pow: Vec<Fr> = vec![Fr::one(); m];
    for i in 0..m - 1 {
      pow[i + 1] = pow[i] * alpha;
    }
    let pow_poly = DensePolynomial { coeffs: domain_m.ifft(&pow) };
    let pow_x2 = util::msm::<G2Projective>(&srs.1, &pow_poly.coeffs);

    // Calculate flat
    let temp:Vec<_> = (0..m).map(|i|inputs[1 + i].g1).collect();
    let flat_x = util::msm::<G1Projective>(&temp, &pow);

    // Check A(x) (A_i = flat_i * v_i)
    let lhs = Bn254::pairing(flat_x, v_x_2) - Bn254::pairing(A.x, srs.1[0]);
    let rhs = Bn254::pairing(A.Q_x, srs.1[n] - srs.1[0]) + Bn254::pairing(C1, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Check v_x_2 is G2 equivalent of v
    let lhs = Bn254::pairing(inputs[0].g1, srs.1[0]);
    let rhs = Bn254::pairing(srs.0[0], v_x_2);
    assert!(lhs == rhs);

    // Check A(x) - A(0) is divisible by x
    let lhs = Bn254::pairing(A.x - A.zero, srs.1[0]);
    let rhs = Bn254::pairing(A.zero_div, srs.1[1]) + Bn254::pairing(C2, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // check B(x) (B_i = w_i * pow_i)
    let lhs = Bn254::pairing(output.g1, pow_x2) - Bn254::pairing(B.x, srs.1[0]);
    let rhs = Bn254::pairing(B.Q_x, srs.1[m] - srs.1[0]) + Bn254::pairing(C3, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);

    // Assume B(0) = A(0)*n/m (which assumes ∑A=∑B)
    let B_zero: G1Affine = (A.zero * (Fr::from(n as u32) * Fr::from(m as u32).inverse().unwrap())).into();

    //check B(x) - B(0) is divisible by x
    let lhs = Bn254::pairing(B.x - B_zero, srs.1[0]);
    let rhs = Bn254::pairing(B.zero_div, srs.1[1]) + Bn254::pairing(C4, srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
  }
}
