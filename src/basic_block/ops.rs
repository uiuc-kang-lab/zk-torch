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
      fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
        assert!(inputs.len() == 1);
        let shape = inputs[0].shape();
        let out = util::array_into_iter(inputs[0])
          .map(|x| {
            let mut x = util::fr_to_int(*x) as f32;
            x /= (1 << self.input_SF) as f32;
            x = $operation(x);
            x *= (1 << self.output_SF) as f32;
            Fr::from(x.round() as i32)
          })
          .collect::<Vec<_>>();

        vec![ArrayD::from_shape_vec(shape, out).unwrap()]
      }
    }
  };
}

make_basic_block!(ExpBasicBlock, { |x: f32| { x.exp() } });
make_basic_block!(LogBasicBlock, { |x: f32| { x.ln() } });
make_basic_block!(ReLUBasicBlock, {
  |x: f32| {
    if x < 0f32 {
      0f32
    } else {
      x
    }
  }
});
make_basic_block!(SqrtBasicBlock, { |x: f32| { x.sqrt() } });
make_basic_block!(ChangeSFBasicBlock, { |x: f32| { x } });
make_basic_block!(ErfBasicBlock, { |x: f32| { util::erf(x) } });
make_basic_block!(SigmoidBasicBlock, { |x: f32| { x.exp() / (1. + x.exp()) } });
make_basic_block!(TanhBasicBlock, { |x: f32| { x.tanh() } });
make_basic_block!(CeilBasicBlock, { |x: f32| { x.ceil() } });
make_basic_block!(NegBasicBlock, { |x: f32| { -x } });
make_basic_block!(CosBasicBlock, { |x: f32| { x.cos() } });
make_basic_block!(SinBasicBlock, { |x: f32| { x.sin() } });
make_basic_block!(TanBasicBlock, { |x: f32| { x.tan() } });
make_basic_block!(ReciprocalBasicBlock, { |x: f32| { 1. / x } });
