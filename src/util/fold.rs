use crate::basic_block::{BasicBlock, CQ2BasicBlock, CQLinBasicBlock};
use ark_bn254::Fr;
use ark_std::Zero;

pub fn get_foldable_bb_info(bb: &Box<dyn BasicBlock>) -> String {
  if bb.is::<CQLinBasicBlock>() {
    let bb = bb.downcast_ref::<CQLinBasicBlock>().unwrap();
    return format!("CQLin-{:?}", bb.setup.shape());
  } else if bb.is::<CQ2BasicBlock>() {
    return "CQ2".to_string();
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

pub fn acc_proof_to_cqlin_acc<P: Clone, Q: Clone>(acc_proof: (&Vec<P>, &Vec<Q>, &Vec<Fr>), log_n: usize, is_prover: bool) -> AccHolder<P, Q> {
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

  let acc_g1_num = if is_prover { 20 } else { 17 };
  let acc_fr_num = if is_prover { log_n + 3 } else { log_n + 1 };
  let acc_err_g2_num = acc_proof.1.len() - 3;

  let err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[acc_g1_num..(acc_g1_num + 5)].to_vec(), acc_proof.1[1..3].to_vec(), vec![]);
  let err5: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[(acc_g1_num + 5)..(acc_g1_num + 8)].to_vec(), vec![], vec![]);
  let err6: (Vec<P>, Vec<Q>, Vec<Fr>) = (acc_proof.0[(acc_g1_num + 8)..(acc_g1_num + 11)].to_vec(), vec![], vec![]);

  let mut errs = vec![err1, err5, err6];
  for i in 0..log_n {
    let err8i = (vec![], vec![], vec![acc_proof.2[acc_fr_num + i]]);
    errs.push(err8i);
  }

  let acc_err1: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 11)..(acc_g1_num + 14 + acc_err_g2_num)].to_vec(),
    acc_proof.1[3..(3 + acc_err_g2_num)].to_vec(),
    vec![],
  );
  let acc_err5: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 14 + acc_err_g2_num)..(acc_g1_num + 17 + acc_err_g2_num)].to_vec(),
    vec![],
    vec![],
  );
  let acc_err6: (Vec<P>, Vec<Q>, Vec<Fr>) = (
    acc_proof.0[(acc_g1_num + 17 + acc_err_g2_num)..(acc_g1_num + 20 + acc_err_g2_num)].to_vec(),
    vec![],
    vec![],
  );

  let mut acc_errs = vec![acc_err1, acc_err5, acc_err6];
  for i in 0..log_n {
    let acc_err8i = (vec![], vec![], vec![acc_proof.2[acc_fr_num + log_n + i]]);
    acc_errs.push(acc_err8i);
  }

  AccHolder {
    acc_g1: acc_proof.0[..acc_g1_num].to_vec(),
    acc_g2: acc_proof.1[..1].to_vec(),
    acc_fr: acc_proof.2[..acc_fr_num].to_vec(),
    mu: acc_proof.2[acc_proof.2.len() - 1],
    errs,
    acc_errs,
  }
}
