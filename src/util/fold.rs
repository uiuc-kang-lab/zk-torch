use crate::basic_block::{BasicBlock, CQ2BasicBlock, CQLinBasicBlock};
use ark_bn254::Fr;

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
