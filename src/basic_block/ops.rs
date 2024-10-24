use super::BasicBlock;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;
use rayon::iter::ParallelIterator;

macro_rules! make_basic_block {
  (
    $name:ident,
    $operation:block
  ) => {
    #[derive(Debug)]
    pub struct $name {
      pub input_SF: usize,
      pub output_SF: usize,
    }
    impl BasicBlock for $name {
      fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
        assert!(inputs.len() == 1);
        let shape = inputs[0].shape();
        let out = util::array_into_iter(inputs[0])
          .map(|x| {
            let mut x = util::fr_to_int(*x) as f64;
            x /= (1 << self.input_SF) as f64;
            x = $operation(x);
            x *= (1 << self.output_SF) as f64;
            Fr::from(x.round() as i128)
          })
          .collect::<Vec<_>>();

        Ok(vec![ArrayD::from_shape_vec(shape, out).unwrap()])
      }
    }
  };
}

make_basic_block!(ExpBasicBlock, { |x: f64| { x.exp() } });
make_basic_block!(LogBasicBlock, { |x: f64| { x.ln() } });
make_basic_block!(ReLUBasicBlock, {
  |x: f64| {
    if x < 0f64 {
      0f64
    } else {
      x
    }
  }
});
make_basic_block!(SqrtBasicBlock, { |x: f64| { x.sqrt() } });
make_basic_block!(ChangeSFBasicBlock, { |x: f64| { x } });
make_basic_block!(ErfBasicBlock, { |x: f64| { util::erf(x) } });
make_basic_block!(SigmoidBasicBlock, { |x: f64| { x.exp() / (1. + x.exp()) } });
make_basic_block!(TanhBasicBlock, { |x: f64| { x.tanh() } });
make_basic_block!(CeilBasicBlock, { |x: f64| { x.ceil() } });
make_basic_block!(NegBasicBlock, { |x: f64| { -x } });
make_basic_block!(CosBasicBlock, { |x: f64| { x.cos() } });
make_basic_block!(SinBasicBlock, { |x: f64| { x.sin() } });
make_basic_block!(TanBasicBlock, { |x: f64| { x.tan() } });
make_basic_block!(ReciprocalBasicBlock, { |x: f64| { 1. / x } });
make_basic_block!(GeLUBasicBlock, { |x: f64| { util::gelu(x) } });
