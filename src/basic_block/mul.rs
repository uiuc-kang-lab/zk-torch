use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::util::{self, acc_proof_to_acc, acc_to_acc_proof, AccHolder, AccProofLayout};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::Mul, ops::Sub, One, UniformRand, Zero};
use ndarray::{azip, ArrayD};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct MulConstBasicBlock {
  pub c: usize,
}

impl AccProofLayout for MulConstBasicBlock {
  fn acc_g1_num(&self, _is_prover: bool) -> usize {
    3
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    0
  }
  fn acc_fr_num(&self, _is_prover: bool) -> usize {
    0
  }
}

impl BasicBlock for MulConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1 && inputs[0].ndim() == 1);
    Ok(vec![inputs[0].map(|x| *x * Fr::from(self.c as u32))])
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let C = srs.X1P[0] * (Fr::from(self.c as u32) * inputs[0].first().unwrap().r - outputs[0].first().unwrap().r);

    let mut proof = vec![C];
    #[cfg(feature = "fold")]
    {
      let inp = inputs[0].first().unwrap();
      let out = outputs[0].first().unwrap();
      let mut additional_g1_for_acc = vec![inp.g1 + srs.Y1P * inp.r, out.g1 + srs.Y1P * out.r];
      proof.append(&mut additional_g1_for_acc);
    }

    return (proof, vec![], Vec::new());
  }

  #[cfg(not(feature = "fold"))]
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    vec![vec![
      (inputs[0].first().unwrap().g1, (srs.X2P[0] * Fr::from(self.c as u32)).into()),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]]
  }

  fn acc_init(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);
    // mu
    acc_proof.2.push(Fr::one());
    acc_proof
  }

  fn acc_prove(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_proof.0.serialize_uncompressed(&mut bytes).unwrap();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let new_acc_proof_g1 = proof.0.iter().zip(acc_proof.0.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    let new_acc_proof_mu = acc_proof.2[0] + acc_gamma;
    (new_acc_proof_g1, Vec::new(), vec![new_acc_proof_mu])
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let mut result = inputs[0].first().unwrap().g1 == proof.0[1] && outputs[0].first().unwrap().g1 == proof.0[2];
    if prev_acc_proof.2.len() == 0 && acc_proof.2[0].is_one() {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_proof.0.serialize_uncompressed(&mut bytes).unwrap();
    proof.0.serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    proof.0.iter().zip(prev_acc_proof.0.iter()).enumerate().for_each(|(i, (x, y))| {
      let z = *x * acc_gamma + *y;
      result &= acc_proof.0[i] == z;
    });
    result &= acc_proof.2[0] == prev_acc_proof.2[0] + acc_gamma;

    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    vec![vec![
      (acc_proof.0[1], (srs.X2P[0] * Fr::from(self.c as u32)).into()),
      (-acc_proof.0[2], srs.X2A[0]),
      (-acc_proof.0[0], srs.Y2A),
    ]]
  }
}

impl AccProofLayout for MulScalarBasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    if is_prover {
      7
    } else {
      4
    }
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    1
  }
  fn acc_fr_num(&self, is_prover: bool) -> usize {
    if is_prover {
      2
    } else {
      0
    }
  }
  fn err_g1_nums_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_g1_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_g2_nums_summable(&self) -> Vec<usize> {
    vec![0]
  }
  fn err_g2_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![0]
  }
}

#[derive(Debug)]
pub struct MulScalarBasicBlock;
impl BasicBlock for MulScalarBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 2 && inputs[0].ndim() <= 1 && inputs[1].len() == 1);
    Ok(vec![inputs[0].map(|x| *x * inputs[1].first().unwrap())])
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0 = &inputs[0].first().unwrap();
    let inp1 = &inputs[1].first().unwrap();
    let out = &outputs[0].first().unwrap();
    let gx2 = srs.X2P[0] * inp1.raw[0] + srs.Y2P * inp1.r;
    let part_C = -srs.X1P[0] * out.r;
    let C = inp0.g1 * inp1.r + inp1.g1 * inp0.r + srs.Y1P * (inp0.r * inp1.r) + part_C;
    let mut proof = vec![C];
    let mut fr = vec![];
    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![inp0.g1 + srs.Y1P * inp0.r, inp1.g1 + srs.Y1P * inp1.r, out.g1 + srs.Y1P * out.r, part_C];
      proof.append(&mut additional_g1_for_acc);
      fr.push(inp0.r);
      fr.push(inp1.r);
    }

    return (proof, vec![gx2], fr);
  }

  #[cfg(not(feature = "fold"))]
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = Vec::new();
    // Verify f(x)*g(x)=h(x)
    checks.push(vec![
      (inputs[0].first().unwrap().g1, proof.1[0]),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], srs.Y2A),
    ]);

    // Verify gx2
    checks.push(vec![(inputs[1].first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -proof.1[0])]);

    checks
  }

  fn acc_init(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof.0[..1].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);

    acc_proof.0.push(inputs[0].first().unwrap().g1);
    acc_proof.0.push(inputs[1].first().unwrap().g1);

    // acc errs and errs
    let g1_zero = G1Projective::zero();
    let g2_zero = G2Projective::zero();
    acc_proof.0.extend(vec![g1_zero; 4 * 2]);
    acc_proof.1.extend(vec![g2_zero; 2 * 2]);

    // mu
    acc_proof.2.push(Fr::one());
    acc_proof
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0_no_blind = &inputs[0].first().unwrap().g1;
    let inp1_no_blind = &inputs[1].first().unwrap().g1;
    let [C, inp0, inp1, out, part_C] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [inp1_2] = proof.1[..] else { panic!("Wrong proof format") };
    let [inp0_r, inp1_r] = proof.2[..] else { panic!("Wrong proof format") };

    let acc_holder = acc_proof_to_acc(self, acc_proof, true);
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };

    let [_acc_C, acc_inp0, _acc_inp1, acc_out, acc_part_C, acc_inp0_no_blind, acc_inp1_no_blind] = acc_holder.acc_g1[..] else {
      panic!("Wrong acc proof format")
    };
    let [acc_inp1_2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    let acc_mu = acc_holder.mu;
    let [acc_inp0_r, acc_inp1_r] = acc_holder.acc_fr[..] else {
      panic!("Wrong acc proof format")
    };

    // Compute the error
    let err: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) = (
      vec![
        inp0,
        acc_inp0,
        acc_out + out * acc_mu,
        acc_part_C
          + part_C * acc_mu
          + acc_inp0_no_blind * inp1_r
          + *inp0_no_blind * acc_inp1_r
          + acc_inp1_no_blind * inp0_r
          + *inp1_no_blind * acc_inp0_r
          + srs.Y1P * (inp0_r * acc_inp1_r + inp1_r * acc_inp0_r),
      ],
      vec![acc_inp1_2, inp1_2],
      vec![],
    );
    let mut errs = vec![err];

    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_holder.acc_g1[1..4].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[1..proof.0.len() - 4].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let proof_0 = vec![C, inp0, inp1, out, part_C, *inp0_no_blind, *inp1_no_blind];
    new_acc_holder.acc_g1 = proof_0.iter().zip(acc_holder.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.acc_g2 = vec![inp1_2 * acc_gamma + acc_inp1_2];
    new_acc_holder.acc_fr = proof.2.iter().zip(acc_holder.acc_fr.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.mu = acc_mu + acc_gamma;
    new_acc_holder.errs = errs.clone();
    new_acc_holder.acc_errs = acc_holder.acc_errs;

    errs[0].0 = errs[0].0.iter().map(|x| (*x * acc_gamma).into()).collect();

    // Append error terms
    let err1_g1_len = new_acc_holder.acc_errs[0].0.len();
    let g_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 2].clone() + errs[0].0[2];
    let c_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 1].clone() + errs[0].0[3];
    let mut errs_0_g1 = errs[0].0[..2].to_vec();
    let mut errs_0_g2 = errs[0].1[..2].to_vec();

    new_acc_holder.acc_errs[0].0 = new_acc_holder.acc_errs[0].0[..err1_g1_len - 2].to_vec();
    new_acc_holder.acc_errs[0].0.append(&mut errs_0_g1);
    new_acc_holder.acc_errs[0].0.push(g_term_g1);
    new_acc_holder.acc_errs[0].0.push(c_term_g1);
    new_acc_holder.acc_errs[0].1.append(&mut errs_0_g2);
    acc_to_acc_proof(new_acc_holder)
  }

  // This function cleans the blinding terms in accumulators for the verifier to do acc_verify without knowing them
  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, true);
    acc_holder.acc_g1[0] = acc_holder.acc_g1[4] * acc_holder.mu
      + acc_holder.acc_g1[5] * acc_holder.acc_fr[1]
      + acc_holder.acc_g1[6] * acc_holder.acc_fr[0]
      + srs.Y1P * acc_holder.acc_fr[0] * acc_holder.acc_fr[1];
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_holder.acc_g1[..acc_holder.acc_g1.len() - 3].to_vec();
    acc_holder.acc_fr = vec![];
    let acc_proof = acc_to_acc_proof(acc_holder);

    // remove blinding terms from bb proof for the verifier
    let cqlin_proof = (vec![proof.0[0].clone()], proof.1.to_vec(), vec![]);

    (
      (
        cqlin_proof.0.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.1.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.2,
      ),
      (
        acc_proof.0.iter().map(|x| (*x).into()).collect(),
        acc_proof.1.iter().map(|x| (*x).into()).collect(),
        acc_proof.2,
      ),
    )
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let inp0 = inputs[0].first().unwrap().g1;
    let inp1 = inputs[1].first().unwrap().g1;
    let out = outputs[0].first().unwrap().g1;
    let mut result = true;

    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0.serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_holder.acc_g1[1..].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[1..].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let proof_0 = vec![inp0, inp1, out];

    proof_0.iter().enumerate().for_each(|(i, x)| {
      let z = *x * acc_gamma + prev_acc_holder.acc_g1[i + 1];
      result &= acc_holder.acc_g1[i + 1] == z;
    });
    result &= acc_holder.acc_g2[0] == prev_acc_holder.acc_g2[0] + proof.1[0] * acc_gamma;
    result &= acc_holder.mu == prev_acc_holder.mu + acc_gamma;
    acc_holder.errs[0].0[acc_holder.errs[0].0.len() - 2..]
      .iter()
      .zip(prev_acc_holder.acc_errs[0].0[prev_acc_holder.acc_errs[0].0.len() - 2..].iter())
      .enumerate()
      .for_each(|(j, (x, y))| {
        let z = *y + *x * acc_gamma;
        result &= z == acc_holder.acc_errs[0].0[acc_holder.acc_errs[0].0.len() - 2 + j];
      });
    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);
    let [acc_C, acc_inp0, acc_inp1, acc_out] = acc_holder.acc_g1[..] else {
      panic!("Wrong acc proof format")
    };
    let [acc_inp1_2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    let acc_mu = acc_holder.mu;
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 1], srs.Y2A));

    let err_1 = temp;
    let mut acc_1: PairingCheck = vec![(acc_inp0, acc_inp1_2), ((-acc_out * acc_mu).into(), srs.X2A[0]), (-acc_C, srs.Y2A)];
    acc_1.extend(err_1);

    let acc_2: PairingCheck = vec![(acc_inp1, srs.X2A[0]), (srs.X1A[0], -acc_inp1_2)];

    vec![acc_1, acc_2]
  }

  fn acc_finalize(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, false);
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 1], srs.Y2A));
    let pairing: Vec<_> = temp.iter().map(|x| x).collect();
    let pairing: (Vec<_>, Vec<_>) = (pairing.iter().map(|x| x.0).collect(), pairing.iter().map(|x| x.1).collect());
    let acc_err1 = Bn254::multi_pairing(pairing.0.iter(), pairing.1.iter());

    acc_holder.errs = vec![];
    acc_holder.acc_errs = vec![];
    let acc_proof = acc_to_acc_proof(acc_holder);
    (acc_proof.0, acc_proof.1, acc_proof.2, vec![acc_err1])
  }
}

impl AccProofLayout for MulBasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {
    if is_prover {
      8
    } else {
      5
    }
  }
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    1
  }
  fn acc_fr_num(&self, is_prover: bool) -> usize {
    if is_prover {
      2
    } else {
      0
    }
  }
  fn err_g1_nums_summable(&self) -> Vec<usize> {
    vec![3]
  }
  fn err_g1_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_g2_nums_summable(&self) -> Vec<usize> {
    vec![0]
  }
  fn err_g2_nums_non_summable(&self) -> Vec<usize> {
    vec![2]
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![0]
  }
}

#[derive(Debug)]
pub struct MulBasicBlock {
  pub len: usize,
}
impl BasicBlock for MulBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[0].shape() == inputs[1].shape());
    let mut r = ArrayD::zeros(inputs[0].dim());
    azip!((r in &mut r, &x in inputs[0], &y in inputs[1]) *r = x * y);
    Ok(vec![r])
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0 = &inputs[0].first().unwrap();
    let inp1 = &inputs[1].first().unwrap();
    let out = &outputs[0].first().unwrap();
    let N = inp0.raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = util::msm::<G2Projective>(&srs.X2A, &inp1.poly.coeffs) + srs.Y2P * inp1.r;
    let t = inp0.poly.mul(&inp1.poly).sub(&out.poly).divide_by_vanishing_poly(domain).unwrap().0;

    // Blinding
    let mut rng = StdRng::from_entropy();
    let r = Fr::rand(&mut rng);
    let tx = util::msm::<G1Projective>(&srs.X1A, &t.coeffs) + srs.Y1P * r;
    let part_C = -(srs.X1P[0] * out.r) - ((srs.X1P[N] - srs.X1P[0]) * r);
    let C = (inp0.g1 * inp1.r) + (inp1.g1 * inp0.r) + (srs.Y1P * (inp0.r * inp1.r)) + part_C;
    let mut proof = vec![tx, C];
    #[cfg(feature = "fold")]
    {
      let mut additional_g1_for_acc = vec![inp0.g1 + srs.Y1P * inp0.r, inp1.g1 + srs.Y1P * inp1.r, out.g1 + srs.Y1P * out.r, part_C];
      proof.append(&mut additional_g1_for_acc);
    }

    return (proof, vec![gx2], Vec::new());
  }

  #[cfg(not(feature = "fold"))]
  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mut checks = vec![];
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    checks.push(vec![
      (inputs[0].first().unwrap().g1, proof.1[0]),
      (-outputs[0].first().unwrap().g1, srs.X2A[0]),
      (-proof.0[0], (srs.X2A[inputs[0].first().unwrap().len] - srs.X2A[0]).into()),
      (-proof.0[1], srs.Y2A),
    ]);
    // Verify gx2
    checks.push(vec![(inputs[1].first().unwrap().g1, srs.X2A[0]), (srs.X1A[0], -proof.1[0])]);
    checks
  }

  fn acc_init(
    &self,
    _srs: &SRS,
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mut acc_proof = (proof.0.clone(), proof.1.clone(), proof.2.clone());
    let inp0_r = &inputs[0].first().unwrap().r;
    let inp1_r = &inputs[1].first().unwrap().r;

    // Fiat-Shamir
    let mut bytes = Vec::new();
    proof.0[..1].serialize_uncompressed(&mut bytes).unwrap();
    util::add_randomness(rng, bytes);
    let _acc_gamma = Fr::rand(rng);

    acc_proof.0.push(inputs[0].first().unwrap().g1);
    acc_proof.0.push(inputs[1].first().unwrap().g1);
    acc_proof.2.push(*inp0_r);
    acc_proof.2.push(*inp1_r);

    // acc errs and errs
    let g1_zero = G1Projective::zero();
    let g2_zero = G2Projective::zero();
    acc_proof.0.extend(vec![g1_zero; 5 * 2]);
    acc_proof.1.extend(vec![g2_zero; 2 * 2]);

    // mu
    acc_proof.2.push(Fr::one());
    acc_proof
  }

  fn acc_prove(
    &self,
    srs: &SRS,
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    _outputs: &Vec<&ArrayD<Data>>,
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let inp0_no_blind = &inputs[0].first().unwrap().g1;
    let inp1_no_blind = &inputs[1].first().unwrap().g1;
    let inp0_r = &inputs[0].first().unwrap().r;
    let inp1_r = &inputs[1].first().unwrap().r;
    let [tx, C, inp0, inp1, out, part_C] = proof.0[..] else {
      panic!("Wrong proof format")
    };

    let [inp1_2] = proof.1[..] else { panic!("Wrong proof format") };

    let acc_holder = acc_proof_to_acc(self, acc_proof, true);
    let mut new_acc_holder = AccHolder {
      acc_g1: Vec::new(),
      acc_g2: Vec::new(),
      acc_fr: Vec::new(),
      mu: Fr::zero(),
      errs: Vec::new(),
      acc_errs: Vec::new(),
    };

    let [acc_tx, _acc_C, acc_inp0, _acc_inp1, acc_out, acc_part_C, acc_inp0_no_blind, acc_inp1_no_blind] = acc_holder.acc_g1[..] else {
      panic!("Wrong acc proof format")
    };
    let [acc_inp1_2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    let acc_mu = acc_holder.mu;
    let [acc_inp0_r, acc_inp1_r] = acc_holder.acc_fr[..] else {
      panic!("Wrong acc proof format")
    };

    // Compute the error
    let err: (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) = (
      vec![
        inp0,
        acc_inp0,
        acc_out + out * acc_mu,
        acc_tx + tx * acc_mu,
        acc_part_C
          + part_C * acc_mu
          + acc_inp0_no_blind * inp1_r
          + *inp0_no_blind * acc_inp1_r
          + acc_inp1_no_blind * inp0_r
          + *inp1_no_blind * acc_inp0_r
          + srs.Y1P * (*inp0_r * acc_inp1_r + *inp1_r * acc_inp0_r),
      ],
      vec![acc_inp1_2, inp1_2],
      vec![],
    );
    let mut errs = vec![err];

    // Fiat-Shamir
    let mut bytes = Vec::new();
    acc_holder.acc_g1[..1].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g1[2..5].serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..1].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let proof_0 = vec![tx, C, inp0, inp1, out, part_C, *inp0_no_blind, *inp1_no_blind];
    new_acc_holder.acc_g1 = proof_0.iter().zip(acc_holder.acc_g1.iter()).map(|(x, y)| *x * acc_gamma + y).collect();
    new_acc_holder.acc_g2 = vec![inp1_2 * acc_gamma + acc_inp1_2];
    let proof_2 = vec![inp0_r, inp1_r];
    new_acc_holder.acc_fr = proof_2.iter().zip(acc_holder.acc_fr.iter()).map(|(x, y)| **x * acc_gamma + y).collect();
    new_acc_holder.mu = acc_mu + acc_gamma;
    new_acc_holder.errs = errs.clone();
    new_acc_holder.acc_errs = acc_holder.acc_errs;

    errs[0].0 = errs[0].0.iter().map(|x| (*x * acc_gamma).into()).collect();

    // Append error terms
    let err1_g1_len = new_acc_holder.acc_errs[0].0.len();
    let g_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 3].clone() + errs[0].0[2];
    let t_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 2].clone() + errs[0].0[3];
    let c_term_g1 = new_acc_holder.acc_errs[0].0[err1_g1_len - 1].clone() + errs[0].0[4];
    let mut errs_0_g1 = errs[0].0[..2].to_vec();
    let mut errs_0_g2 = errs[0].1[..2].to_vec();

    new_acc_holder.acc_errs[0].0 = new_acc_holder.acc_errs[0].0[..err1_g1_len - 3].to_vec();
    new_acc_holder.acc_errs[0].0.append(&mut errs_0_g1);
    new_acc_holder.acc_errs[0].0.push(g_term_g1);
    new_acc_holder.acc_errs[0].0.push(t_term_g1);
    new_acc_holder.acc_errs[0].0.push(c_term_g1);
    new_acc_holder.acc_errs[0].1.append(&mut errs_0_g2);
    acc_to_acc_proof(new_acc_holder)
  }

  fn acc_clean(
    &self,
    srs: &SRS,
    proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
    acc_proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>),
  ) -> ((Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>), (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, true);
    //vec![tx, C, inp0, inp1, out, part_C, *inp0_no_blind, *inp1_no_blind]
    acc_holder.acc_g1[1] = acc_holder.acc_g1[5] * acc_holder.mu
      + acc_holder.acc_g1[6] * acc_holder.acc_fr[1]
      + acc_holder.acc_g1[7] * acc_holder.acc_fr[0]
      + srs.Y1P * acc_holder.acc_fr[0] * acc_holder.acc_fr[1];
    // remove blinding terms from acc proof for the verifier
    acc_holder.acc_g1 = acc_holder.acc_g1[..acc_holder.acc_g1.len() - 3].to_vec();
    acc_holder.acc_fr = vec![];
    let acc_proof = acc_to_acc_proof(acc_holder);

    // remove blinding terms from bb proof for the verifier
    let cqlin_proof = (proof.0[..2].to_vec(), proof.1.to_vec(), vec![]);

    (
      (
        cqlin_proof.0.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.1.iter().map(|x| (*x).into()).collect(),
        cqlin_proof.2,
      ),
      (
        acc_proof.0.iter().map(|x| (*x).into()).collect(),
        acc_proof.1.iter().map(|x| (*x).into()).collect(),
        acc_proof.2,
      ),
    )
  }

  fn acc_verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    prev_acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Option<bool> {
    let inp0 = inputs[0].first().unwrap().g1;
    let inp1 = inputs[1].first().unwrap().g1;
    let out = outputs[0].first().unwrap().g1;
    let mut result = true;

    let prev_acc_holder = acc_proof_to_acc(self, prev_acc_proof, false);
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);

    if prev_acc_holder.mu.is_zero() && acc_holder.mu.is_one() {
      // skip verifying RLC because no RLC was done in acc_init.
      // Fiat-shamir
      let mut bytes = Vec::new();
      proof.0[..1].serialize_uncompressed(&mut bytes).unwrap();
      util::add_randomness(rng, bytes);
      let _acc_gamma = Fr::rand(rng);
      return Some(result);
    }

    // Fiat-Shamir
    let mut bytes = Vec::new();
    prev_acc_holder.acc_g1[..1].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g1[2..5].serialize_uncompressed(&mut bytes).unwrap();
    prev_acc_holder.acc_g2.serialize_uncompressed(&mut bytes).unwrap();
    proof.0[..1].serialize_uncompressed(&mut bytes).unwrap();
    proof.1.serialize_uncompressed(&mut bytes).unwrap();
    acc_holder.errs.iter().for_each(|(g1, g2, f)| {
      g1.serialize_uncompressed(&mut bytes).unwrap();
      g2.serialize_uncompressed(&mut bytes).unwrap();
      f.serialize_uncompressed(&mut bytes).unwrap();
    });
    util::add_randomness(rng, bytes);
    let acc_gamma = Fr::rand(rng);

    let proof_0 = vec![proof.0[0], proof.0[1], inp0, inp1, out];

    proof_0.iter().enumerate().for_each(|(i, x)| {
      if i != 1 {
        // i==1 is C
        let z = *x * acc_gamma + prev_acc_holder.acc_g1[i];
        result &= acc_holder.acc_g1[i] == z;
      }
    });
    result &= acc_holder.acc_g2[0] == prev_acc_holder.acc_g2[0] + proof.1[0] * acc_gamma;
    result &= acc_holder.mu == prev_acc_holder.mu + acc_gamma;
    acc_holder.errs[0].0[acc_holder.errs[0].0.len() - 3..]
      .iter()
      .zip(prev_acc_holder.acc_errs[0].0[prev_acc_holder.acc_errs[0].0.len() - 3..].iter())
      .enumerate()
      .for_each(|(j, (x, y))| {
        let z = *y + *x * acc_gamma;
        result &= z == acc_holder.acc_errs[0].0[acc_holder.acc_errs[0].0.len() - 3 + j];
      });
    Some(result)
  }

  fn acc_decide(&self, srs: &SRS, acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> Vec<PairingCheck> {
    let acc_holder = acc_proof_to_acc(self, acc_proof, false);
    let [acc_tx, acc_C, acc_inp0, acc_inp1, acc_out] = acc_holder.acc_g1[..] else {
      panic!("Wrong acc proof format")
    };
    let [acc_inp1_2] = acc_holder.acc_g2[..] else {
      panic!("Wrong acc proof format")
    };
    let acc_mu = acc_holder.mu;
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 1], (srs.X2A[self.len] - srs.X2A[0]).into()));
    temp.push((err_1.0[err_1.1.len() + 2], srs.Y2A));

    let err_1 = temp;
    let mut acc_1: PairingCheck = vec![
      (acc_inp0, acc_inp1_2),
      ((-acc_out * acc_mu).into(), srs.X2A[0]),
      ((-acc_tx * acc_mu).into(), (srs.X2A[self.len] - srs.X2A[0]).into()),
      (-acc_C, srs.Y2A),
    ];
    acc_1.extend(err_1);

    let acc_2: PairingCheck = vec![(acc_inp1, srs.X2A[0]), (srs.X1A[0], -acc_inp1_2)];

    vec![acc_1, acc_2]
  }

  fn acc_finalize(
    &self,
    srs: &SRS,
    acc_proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
  ) -> (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
    let mut acc_holder = acc_proof_to_acc(self, acc_proof, false);
    let err_1 = &acc_holder.acc_errs[0];

    let mut temp: PairingCheck = vec![];
    for i in 0..err_1.1.len() {
      temp.push((-err_1.0[i], err_1.1[i]));
    }
    temp.push((err_1.0[err_1.1.len()], srs.X2A[0]));
    temp.push((err_1.0[err_1.1.len() + 1], (srs.X2A[self.len] - srs.X2A[0]).into()));
    temp.push((err_1.0[err_1.1.len() + 2], srs.Y2A));
    let pairing: Vec<_> = temp.iter().map(|x| x).collect();
    let pairing: (Vec<_>, Vec<_>) = (pairing.iter().map(|x| x.0).collect(), pairing.iter().map(|x| x.1).collect());
    let acc_err1 = Bn254::multi_pairing(pairing.0.iter(), pairing.1.iter());

    acc_holder.errs = vec![];
    acc_holder.acc_errs = vec![];
    let acc_proof = acc_to_acc_proof(acc_holder);
    (acc_proof.0, acc_proof.1, acc_proof.2, vec![acc_err1])
  }
}
