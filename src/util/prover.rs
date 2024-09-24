/*
 * Prover utilities:
 * The functions are used for proving-related operations, such as
 * generating CQ tables and converting them to Data (generating commitment).
 */
use crate::basic_block::{BasicBlock, Data, SRS};
use crate::{onnx, util};
use ark_bn254::{Fr, G1Projective};
use ark_std::Zero;
use ndarray::{arr0, arr1, concatenate, Array1, ArrayD, Axis, IxDyn};
use rayon::range;

#[derive(Debug, Clone, PartialEq)]
pub enum CQArrayType {
  Negative,
  NonNegative,
  NonZero,
  NonPositive,
  Positive,
  Custom(Vec<Fr>),
}

pub fn gen_cq_array(cq_type: CQArrayType) -> ArrayD<Fr> {
  let r = match cq_type {
    CQArrayType::Negative => (*onnx::CQ_RANGE_LOWER..0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonNegative => (0..*onnx::CQ_RANGE as i32).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonZero => (*onnx::CQ_RANGE_LOWER..-*onnx::CQ_RANGE_LOWER + 1).filter(|&x| x != 0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonPositive => (*onnx::CQ_RANGE_LOWER + 1..1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Positive => (1..-*onnx::CQ_RANGE_LOWER + 1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Custom(range) => range,
  };
  arr1(&r).into_dyn()
}

pub fn check_cq_array(cq_type: CQArrayType, x_int: i32) -> bool {
  let result = match cq_type {
    CQArrayType::Negative => x_int < 0 && x_int >= *onnx::CQ_RANGE_LOWER,
    CQArrayType::NonNegative => x_int >= 0 && x_int < (*onnx::CQ_RANGE as i32),
    CQArrayType::NonZero => x_int != 0 && x_int >= *onnx::CQ_RANGE_LOWER && x_int <= -*onnx::CQ_RANGE_LOWER,
    CQArrayType::NonPositive => x_int <= 0 && x_int > *onnx::CQ_RANGE_LOWER,
    CQArrayType::Positive => x_int > 0 && x_int <= -*onnx::CQ_RANGE_LOWER,
    CQArrayType::Custom(range) => {
      let range = range.iter().map(|x| util::fr_to_int(*x)).collect::<Vec<_>>();
      range.contains(&x_int)
    }
  };
  if !result {
    println!("{:?}", x_int);
  }
  result
}

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, offset: i32, size: usize) -> ArrayD<Fr> {
  let range = Array1::from_shape_fn(size, |i| Fr::from(i as u32) + Fr::from(offset)).into_dyn();
  let result = &(**basic_block).run(&ArrayD::zeros(IxDyn(&[0])), &vec![&range]).unwrap()[0];
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
