/*
 * Prover utilities:
 * The functions are used for proving-related operations, such as
 * generating CQ tables and converting them to Data (generating commitment).
 */
use crate::basic_block::{BasicBlock, Data, SRS};
use ark_bn254::{Fr, G1Projective};
use ark_std::Zero;
use ndarray::{arr0, concatenate, Array1, ArrayD, Axis, IxDyn};

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, offset: i32, size: usize) -> ArrayD<Fr> {
  let range = Array1::from_shape_fn(size, |i| Fr::from(i as u32) + Fr::from(offset)).into_dyn();
  let result = &(**basic_block).run(&ArrayD::zeros(IxDyn(&[0])), &vec![&range])[0];
  let range = range.view().into_shape(IxDyn(&[1, size])).unwrap();
  let result = result.view().into_shape(IxDyn(&[1, size])).unwrap();
  concatenate(Axis(0), &[range, result]).unwrap()
}

pub fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() <= 1 {
    return arr0(Data::new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  let mut a = a.map_axis(Axis(a.ndim() - 1), |r| Data {
    raw: r.as_slice().unwrap().to_vec(),
    poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
    r: Fr::zero(),
    g1: G1Projective::zero(),
  });
  a.par_map_inplace(|x| {
    *x = Data::new(srs, &x.raw);
  });
  a
}
