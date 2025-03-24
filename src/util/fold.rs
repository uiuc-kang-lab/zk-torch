use crate::basic_block::*;
use crate::util::get_cq_N;
use ark_bn254::Fr;
use ark_std::Zero;

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

pub struct AccHolder<P: Clone, Q: Clone> {
  pub acc_g1: Vec<P>,
  pub acc_g2: Vec<Q>,
  pub acc_fr: Vec<Fr>,
  pub mu: Fr,
  pub errs: Vec<(Vec<P>, Vec<Q>, Vec<Fr>)>,     // i-th element contains err_i: [e_j]_j=1..n
  pub acc_errs: Vec<(Vec<P>, Vec<Q>, Vec<Fr>)>, // i-th element contains acc_err_i += SUM{acc_gamma^j * e_j} for j=1..n
}

pub fn acc_to_acc_proof<P: Clone, Q: Clone>(acc: AccHolder<P, Q>) -> (Vec<P>, Vec<Q>, Vec<Fr>) {
  let mut g1 = acc.acc_g1.clone();
  let mut g2 = acc.acc_g2.clone();
  let mut fr = acc.acc_fr.clone();
  acc.errs.iter().for_each(|x| {
    g1.extend(x.0.clone());
    g2.extend(x.1.clone());
    fr.extend(x.2.clone());
  });
  acc.acc_errs.iter().for_each(|x| {
    g1.extend(x.0.clone());
    g2.extend(x.1.clone());
    fr.extend(x.2.clone());
  });
  fr.push(acc.mu);
  (g1, g2, fr)
}

pub trait AccProofLayout: BasicBlock {
  fn acc_g1_num(&self, is_prover: bool) -> usize;
  fn acc_g2_num(&self, is_prover: bool) -> usize;
  fn acc_fr_num(&self, is_prover: bool) -> usize;
  fn err_g1_nums_summable(&self) -> Vec<usize> {
    vec![]
  }
  fn err_g1_nums_non_summable(&self) -> Vec<usize> {
    vec![]
  }
  fn err_g2_nums_summable(&self) -> Vec<usize> {
    vec![]
  }
  fn err_g2_nums_non_summable(&self) -> Vec<usize> {
    vec![]
  }
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![]
  }
}

pub fn acc_proof_to_acc<P: Clone, Q: Clone>(bb: &dyn AccProofLayout, acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>), is_prover: bool) -> AccHolder<P, Q> {
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
  let (mut err_g1_start, mut err_g2_start, mut err_fr_start) = (acc_g1_num, acc_g2_num, acc_fr_num);
  for i in 0..bb.err_g1_nums_summable().len() {
    let (err_g1_end, err_g2_end, err_fr_end) = (
      err_g1_start + bb.err_g1_nums_summable()[i] + bb.err_g1_nums_non_summable()[i],
      err_g2_start + bb.err_g2_nums_summable()[i] + bb.err_g2_nums_non_summable()[i],
      err_fr_start + bb.err_fr_nums()[i],
    );
    let err: (Vec<P>, Vec<Q>, Vec<Fr>) = (
      acc_proof.0[err_g1_start..err_g1_end].to_vec(),
      acc_proof.1[err_g2_start..err_g2_end].to_vec(),
      acc_proof.2[err_fr_start..err_fr_end].to_vec(),
    );
    err_g1_start = err_g1_end;
    err_g2_start = err_g2_end;
    err_fr_start = err_fr_end;

    errs.push(err);
  }

  let acc_err_g2_num = acc_proof.1.len() - err_g2_start;

  let mut acc_errs = vec![];
  let acc_times = if acc_err_g2_num == 0 {
    0
  } else {
    (acc_err_g2_num - bb.err_g2_nums_summable().iter().sum::<usize>()) / bb.err_g2_nums_non_summable().iter().sum::<usize>()
  };
  for i in 0..bb.err_g1_nums_summable().len() {
    let (err_g1_end, err_g2_end, err_fr_end) = (
      err_g1_start + bb.err_g1_nums_summable()[i] + bb.err_g1_nums_non_summable()[i] * acc_times,
      err_g2_start + bb.err_g2_nums_summable()[i] + bb.err_g2_nums_non_summable()[i] * acc_times,
      err_fr_start + bb.err_fr_nums()[i],
    );

    let acc_err: (Vec<P>, Vec<Q>, Vec<Fr>) = (
      acc_proof.0[err_g1_start..err_g1_end].to_vec(),
      acc_proof.1[err_g2_start..err_g2_end].to_vec(),
      acc_proof.2[err_fr_start..err_fr_end].to_vec(),
    );
    err_g1_start = err_g1_end;
    err_g2_start = err_g2_end;
    err_fr_start = err_fr_end;

    acc_errs.push(acc_err);
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
