use crate::basic_block::*;
use crate::util::get_cq_N;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::Zero;
use ark_ec::AffineRepr;
use rand::rngs::StdRng;

pub fn get_foldable_bb_info(bb: &Box<dyn BasicBlock>) -> String {
  if bb.is::<CQLinBasicBlock>() {
    let bb = bb.downcast_ref::<CQLinBasicBlock>().unwrap();
    return format!("CQLin-{:?}", bb.setup.shape());
  } else if bb.is::<CQ2BasicBlock>() {
    let bb = bb.downcast_ref::<CQ2BasicBlock>().unwrap();
    return format!("CQ2-{}-{}", bb.n, bb.setup.as_ref().unwrap().2);
  } else if bb.is::<CQBasicBlock>() {
    let bb = bb.downcast_ref::<CQBasicBlock>().unwrap();
    return format!("CQ-{}-{}", bb.n, get_cq_N(&bb.setup));
  } else if bb.is::<RepeaterBasicBlock>() {
    let bb = bb.downcast_ref::<RepeaterBasicBlock>().unwrap();
    let b = &bb.basic_block;
    return format!("Repeater-{}", get_foldable_bb_info(b));
  } else {
    return format!("{:?}", bb);
  }
}

#[derive(Clone, Debug)]
pub struct AccHolder<P: Copy, Q: Copy> {
  pub acc_g1: Vec<P>,
  pub acc_g2: Vec<Q>,
  pub acc_fr: Vec<Fr>,
  pub mu: Fr,
  pub errs: Vec<(Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)>, // i-th element contains err_i: [e_j]_j=1..n
  pub acc_errs: Vec<(Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>)>, // i-th element contains acc_err_i += SUM{acc_gamma^j * e_j} for j=1..n
}

pub fn acc_to_acc_proof<P: Copy, Q: Copy>(acc: AccHolder<P, Q>) -> (Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
  if acc.acc_g1.len() == 0 && acc.acc_g2.len() == 0 && acc.acc_fr.len() == 0 {
    return (vec![], vec![], vec![], vec![]);
  }
  let mut g1 = acc.acc_g1;
  let mut g2 = acc.acc_g2;
  let mut fr = acc.acc_fr;
  let mut gt = vec![];
  acc.errs.into_iter().for_each(|x| {
    g1.extend(x.0);
    g2.extend(x.1);
    fr.extend(x.2);
    gt.extend(x.3);
  });
  acc.acc_errs.into_iter().for_each(|x| {
    g1.extend(x.0);
    g2.extend(x.1);
    fr.extend(x.2);
    gt.extend(x.3);
  });
  fr.push(acc.mu);
  (g1, g2, fr, gt)
}

pub trait AccProofLayout: BasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize {0}
  fn acc_g2_num(&self, is_prover: bool) -> usize {0}
  fn acc_fr_num(&self, is_prover: bool) -> usize {0}
  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective>;
  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine>;
  fn mira_prove(
    &self,
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective>;
  fn mira_verify(
    &self,
    acc_1: AccHolder<G1Affine, G2Affine>,
    acc_2: AccHolder<G1Affine, G2Affine>,
    new_acc: AccHolder<G1Affine, G2Affine>,
    rng: &mut StdRng,
  ) -> Option<bool> {
    None
  }
  fn err_g1_nums(&self) -> Vec<usize> {
    vec![]
  }
  fn err_g2_nums(&self) -> Vec<usize> {
    vec![]
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![]
  }
  fn err_gt_nums(&self) -> Vec<usize> {
    vec![]
  }
  fn prover_dummy_holder(&self) -> AccHolder<G1Projective, G2Projective> {
    let errs: Vec<_> = self.err_g1_nums().iter().enumerate().map(|(i, v)| (vec![G1Projective::zero(); *v], vec![G2Projective::zero(); self.err_g2_nums()[i]], vec![Fr::zero(); self.err_fr_nums()[i]], vec![PairingOutput::<Bn<ark_bn254::Config>>::zero(); self.err_gt_nums()[i]])).collect();
    AccHolder {
      acc_g1: vec![G1Projective::zero(); self.acc_g1_num(true)],
      acc_g2: vec![G2Projective::zero(); self.acc_g2_num(true)],
      acc_fr: vec![Fr::zero(); self.acc_fr_num(true)],
      mu: Fr::zero(),
      errs: errs.clone(),
      acc_errs: errs
    }
  }
  fn verifier_dummy_holder(&self) -> AccHolder<G1Affine, G2Affine> {
    let errs: Vec<_> = self.err_g1_nums().iter().enumerate().map(|(i, v)| (vec![G1Affine::zero(); *v], vec![G2Affine::zero(); self.err_g2_nums()[i]], vec![Fr::zero(); self.err_fr_nums()[i]], vec![PairingOutput::<Bn<ark_bn254::Config>>::zero(); self.err_gt_nums()[i]])).collect();
    AccHolder {
      acc_g1: vec![G1Affine::zero(); self.acc_g1_num(false)],
      acc_g2: vec![G2Affine::zero(); self.acc_g2_num(false)],
      acc_fr: vec![Fr::zero(); self.acc_fr_num(false)],
      mu: Fr::zero(),
      errs: errs.clone(),
      acc_errs: errs
    }
  }
}

pub fn acc_proof_to_acc<P: Copy, Q: Copy>(
  bb: &dyn AccProofLayout,
  acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>, &Vec<PairingOutput<Bn<ark_bn254::Config>>>),
  is_prover: bool,
) -> AccHolder<P, Q> {
  if acc_proof.0.len() == 0 && acc_proof.1.len() == 0 && acc_proof.2.len() == 0 {
    return AccHolder {
      acc_g1: vec![],
      acc_g2: vec![],
      acc_fr: vec![],
      mu: Fr::zero(),
      errs: vec![],
      acc_errs: vec![],
    };
  }

  let acc_g1_num = bb.acc_g1_num(is_prover);
  let acc_g2_num = bb.acc_g2_num(is_prover);
  let acc_fr_num = bb.acc_fr_num(is_prover);

  let mut errs = vec![];
  let (mut err_g1_start, mut err_g2_start, mut err_fr_start, mut err_gt_start) = (acc_g1_num, acc_g2_num, acc_fr_num, 0);
  for i in 0..bb.err_g1_nums().len() {
    let (err_g1_end, err_g2_end, err_fr_end, err_gt_end) = (
      err_g1_start + bb.err_g1_nums()[i],
      err_g2_start + bb.err_g2_nums()[i],
      err_fr_start + bb.err_fr_nums()[i],
      err_gt_start + bb.err_gt_nums()[i],
    );
    let err: (Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) = (
      acc_proof.0[err_g1_start..err_g1_end].to_vec(),
      acc_proof.1[err_g2_start..err_g2_end].to_vec(),
      acc_proof.2[err_fr_start..err_fr_end].to_vec(),
      acc_proof.3[err_gt_start..err_gt_end].to_vec(),
    );
    err_g1_start = err_g1_end;
    err_g2_start = err_g2_end;
    err_fr_start = err_fr_end;
    err_gt_start = err_gt_end;

    errs.push(err);
  }
  let mut acc_errs = vec![];
  for i in 0..bb.err_g1_nums().len() {
    let (err_g1_end, err_g2_end, err_fr_end, err_gt_end) = (
      err_g1_start + bb.err_g1_nums()[i],
      err_g2_start + bb.err_g2_nums()[i],
      err_fr_start + bb.err_fr_nums()[i],
      err_gt_start + bb.err_gt_nums()[i],
    );
    let err: (Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) = (
      acc_proof.0[err_g1_start..err_g1_end].to_vec(),
      acc_proof.1[err_g2_start..err_g2_end].to_vec(),
      acc_proof.2[err_fr_start..err_fr_end].to_vec(),
      acc_proof.3[err_gt_start..err_gt_end].to_vec(),
    );
    err_g1_start = err_g1_end;
    err_g2_start = err_g2_end;
    err_fr_start = err_fr_end;
    err_gt_start = err_gt_end;

    acc_errs.push(err);
  }

  AccHolder {
    acc_g1: acc_proof.0[..acc_g1_num].to_vec(),
    acc_g2: acc_proof.1[..acc_g2_num].to_vec(),
    acc_fr: acc_proof.2[..acc_fr_num].to_vec(),
    mu: acc_proof.2[acc_proof.2.len() - 1],
    errs,
    acc_errs,
  }
}
