use crate::basic_block::*;
use crate::util::get_cq_N;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::Bn;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::AffineRepr;
use ark_std::Zero;
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

// holder_to_acc_proof converts an accumulator holder (AccHolder) to an accumulator proof
pub fn holder_to_acc_proof<P: Copy, Q: Copy>(acc: AccHolder<P, Q>) -> (Vec<P>, Vec<Q>, Vec<Fr>, Vec<PairingOutput<Bn<ark_bn254::Config>>>) {
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

// AccProofLayout is a trait that defines the layout of the accumulator proof
// It is used to implement the generalized accumulator proof for different basic blocks
pub trait AccProofLayout: BasicBlock {
  // acc_g1_num returns the number of G1 elements in an accumulator instance
  fn acc_g1_num(&self, _is_prover: bool) -> usize {
    0
  }

  // acc_g2_num returns the number of G2 elements in an accumulator instance
  fn acc_g2_num(&self, _is_prover: bool) -> usize {
    0
  }

  // acc_fr_num returns the number of Fr elements in an accumulator instance
  fn acc_fr_num(&self, _is_prover: bool) -> usize {
    0
  }

  // prover_proof_to_acc converts the NARK proof from the prover to an accumulator instance
  fn prover_proof_to_acc(&self, proof: (&Vec<G1Projective>, &Vec<G2Projective>, &Vec<Fr>)) -> AccHolder<G1Projective, G2Projective>;

  // verifier_proof_to_acc converts the NARK proof from the verifier to an accumulator instance
  fn verifier_proof_to_acc(&self, proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)) -> AccHolder<G1Affine, G2Affine>;

  // mira_prove is the main function that performs the generalized accumulator proof
  fn mira_prove(
    &self,
    srs: &SRS,
    acc_1: AccHolder<G1Projective, G2Projective>,
    acc_2: AccHolder<G1Projective, G2Projective>,
    rng: &mut StdRng,
  ) -> AccHolder<G1Projective, G2Projective>;

  // mira_verify is the main function that verifies the generalized accumulator proof
  fn mira_verify(
    &self,
    _acc_1: AccHolder<G1Affine, G2Affine>,
    _acc_2: AccHolder<G1Affine, G2Affine>,
    _new_acc: AccHolder<G1Affine, G2Affine>,
    _rng: &mut StdRng,
  ) -> Option<bool> {
    None
  }

  // err_g1_nums returns the number of G1 elements in the error terms
  // Its length should be equal to the number of error terms
  // The i-th element of the vector is the number of G1 elements in the i-th error term
  // Note: technically, we can realize the pairing in G1 with the predefined G2 elements;
  // but we do not do this in the current implementation because it is not efficient
  fn err_g1_nums(&self) -> Vec<usize> {
    vec![]
  }

  // err_g2_nums returns the number of G2 elements in the error terms
  // Its length should be equal to the number of error terms
  // The i-th element of the vector is the number of G2 elements in the i-th error term
  // Note: technically, we can realize the pairing in G2 with the predefined G1 elements;
  // but we do not do this in the current implementation because it is not efficient
  fn err_g2_nums(&self) -> Vec<usize> {
    vec![]
  }

  // err_fr_nums returns the number of Fr elements in the error terms
  // Its length should be equal to the number of error terms
  // The i-th element of the vector is the number of Fr elements in the i-th error term
  fn err_fr_nums(&self) -> Vec<usize> {
    vec![]
  }

  // err_gt_nums returns the number of GT elements (the realized pairings) in the error terms
  // Its length should be equal to the number of error terms
  // The i-th element of the vector is the number of GT elements in the i-th error term
  // Note: we only realize the pairing between G1 and G2 elements when both of them are not predefined
  // In this case, we do the pairing (i.e., gt = e(g1, g2)) to prevent too much memory usage
  fn err_gt_nums(&self) -> Vec<usize> {
    vec![]
  }

  fn prover_dummy_holder(&self) -> AccHolder<G1Projective, G2Projective> {
    let errs: Vec<_> = self
      .err_g1_nums()
      .iter()
      .enumerate()
      .map(|(i, v)| {
        (
          vec![G1Projective::zero(); *v],
          vec![G2Projective::zero(); self.err_g2_nums()[i]],
          vec![Fr::zero(); self.err_fr_nums()[i]],
          vec![PairingOutput::<Bn<ark_bn254::Config>>::zero(); self.err_gt_nums()[i]],
        )
      })
      .collect();
    AccHolder {
      acc_g1: vec![G1Projective::zero(); self.acc_g1_num(true)],
      acc_g2: vec![G2Projective::zero(); self.acc_g2_num(true)],
      acc_fr: vec![Fr::zero(); self.acc_fr_num(true)],
      mu: Fr::zero(),
      errs: errs.clone(),
      acc_errs: errs,
    }
  }

  fn verifier_dummy_holder(&self) -> AccHolder<G1Affine, G2Affine> {
    let errs: Vec<_> = self
      .err_g1_nums()
      .iter()
      .enumerate()
      .map(|(i, v)| {
        (
          vec![G1Affine::zero(); *v],
          vec![G2Affine::zero(); self.err_g2_nums()[i]],
          vec![Fr::zero(); self.err_fr_nums()[i]],
          vec![PairingOutput::<Bn<ark_bn254::Config>>::zero(); self.err_gt_nums()[i]],
        )
      })
      .collect();
    AccHolder {
      acc_g1: vec![G1Affine::zero(); self.acc_g1_num(false)],
      acc_g2: vec![G2Affine::zero(); self.acc_g2_num(false)],
      acc_fr: vec![Fr::zero(); self.acc_fr_num(false)],
      mu: Fr::zero(),
      errs: errs.clone(),
      acc_errs: errs,
    }
  }
}

// acc_proof_to_holder converts an accumulator proof to an accumulator holder (AccHolder)
// L: ?Sized means "L could be a type that implements Sized or a type that doesn’t implement Sized".
// This is required because we need this property to implement get_acc_proof_bases(.) in repeater.rs
pub fn acc_proof_to_holder<P: Copy, Q: Copy, L: AccProofLayout + ?Sized>(
  bb: &L,
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

// Define the accumulator terms
//  The first argument is the name of the accumulator term
//  The second argument is the list of public accumulator terms
//  The third argument is the list of prover-only accumulator terms
//  The function idx returns the index of the accumulator term
//  The index can be used to access the element in the accumulator holder
// Example:
//   define_acc_terms!(BasicBlockG1Terms, [pub_A, pub_B], [priv_A]);
//   where BasicBlockG1Terms is the name of the accumulator term
//   pub_A and pub_B are the two public accumulator terms
//   priv_A is the accumulator term only available to the prover
//   BasicBlockG1Terms::idx(BasicBlockG1Terms::pub_A) = 0
//   BasicBlockG1Terms::idx(BasicBlockG1Terms::pub_B) = 1
//   BasicBlockG1Terms::idx(BasicBlockG1Terms::priv_A) = 2
//   To get pub_A, use acc_holder.acc_g1[BasicBlockG1Terms::idx(BasicBlockG1Terms::pub_A)]
//   To get pub_B and priv_A, the logic is the same
#[macro_export]
macro_rules! define_acc_terms {
  ($name:ident, [$($public:ident),*], [$($prover:ident),*]) => {
    #[derive(Debug, Clone, Copy)]
    pub struct $name<T: Copy> {
      #[allow(dead_code)]
      phantom: std::marker::PhantomData<T>,
      $(pub $public: T,)*
      $(pub $prover: Option<T>,)*
    }

    impl<T: Copy> $name<T> {
      pub const PUBLIC_COUNT: usize = 0 $(+ { let _ = stringify!($public); 1 })*;
      pub const PROVER_COUNT: usize = 0 $(+ { let _ = stringify!($prover); 1 })*;
      pub const COUNT: usize = Self::PUBLIC_COUNT + Self::PROVER_COUNT;
      pub fn from_vec(vec: &[T]) -> Self {
        #[allow(unused_mut)]
        let mut iter = vec.iter();
        if vec.len() == Self::COUNT {
          Self {
            phantom: std::marker::PhantomData,
            $($public: *iter.next().unwrap(),)*
            $($prover: Some(*iter.next().unwrap()),)*
          }
        } else if vec.len() == Self::PUBLIC_COUNT {
          Self {
            phantom: std::marker::PhantomData,
            $($public: *iter.next().unwrap(),)*
            $($prover: None,)*
          }
        } else {
          panic!("Invalid vector length");
        }
      }

      pub fn to_vec(&self) -> Vec<T> {
        #[allow(unused_mut)]
        let mut vec = vec![];
        $(vec.push(self.$public);)*
        $(if let Some(prover) = self.$prover { vec.push(prover); })*
        vec
      }
    }
  };
}

// Define the error terms in the accumulator proof
// The first argument is the name of the error term
// The i-th argument is the list of terms in the i-th error (i > 0)
// Example:
//   define_acc_err_terms!(BasicBlockErrG1Terms, [Err_A], [Err_B, Err_C], []);
//   where BasicBlockErrG1Terms is the name of the error term
//   Note there are three []s in the definition, which means there are three error terms for this BasicBlock
//   Err_A is the 1st error term that contains A
//   Err_B and Err_C are the 2nd error term that contain B and C
//   The 3rd error term contains nothing, but it is still needed to be specified as there may exist GT terms in the error term
//   To get Err_A, use the following code
//      let (group, idx) = BasicBlockErrG1Terms::idx(BasicBlockErrG1Terms::Err_A);
//      let err_A = acc_holder.errs[group].0[idx];
//   To get Err_B and Err_C, the logic is the same
//   If there is no error term, just use define_acc_err_terms!(BasicBlockErrG1Terms); // (i.e., no arguments)
#[macro_export]
macro_rules! define_acc_err_terms {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {}

        impl $name {
            pub const COUNTS: &'static [usize] = &[];
            pub const COUNT: usize = 0;

            pub fn idx(_idx: Self) -> (usize, usize) {
                panic!("Index out of bounds: empty error terms");
            }
        }
    };

    ($name:ident, $([$($group:ident),*]),+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($(
                $group,
            )*)+
        }

        impl $name {
            pub const COUNTS: &'static [usize] = &[
                $(0 $(+ { let _ = $name::$group; 1 })*),+
            ];

            pub const COUNT: usize = 0 $($(+ { let _ = $name::$group; 1 })*)+;

            pub fn idx(idx: Self) -> (usize, usize) {
                let mut sum = 0;
                let idx = idx as usize;
                for (i, &count) in Self::COUNTS.iter().enumerate() {
                    sum += count;
                    if idx < sum {
                        return (i, idx - sum + count);
                    }
                }
                panic!("Index out of bounds");
            }
        }
    };
}
