use super::BasicBlock;
use crate::basic_block::BasicBlockType;
use crate::util;
use ark_bn254::Fr;
use ndarray::ArrayD;

pub fn is_nonlinearity(block_type: BasicBlockType) -> bool {
  match block_type {
    BasicBlockType::ChangeSF | BasicBlockType::Exp | BasicBlockType::Log | BasicBlockType::ReLU | BasicBlockType::Sqrt => true,
    _ => false,
  }
}

macro_rules! make_basic_block {
  (
    $name:ident,
    $block_name:ident,
    $operation:block
  ) => {
    #[derive(Debug)]
    pub struct $block_name {
      pub input_SF: usize,
      pub output_SF: usize,
    }

    impl BasicBlock for $block_name {
      fn block_type(&self) -> BasicBlockType {
        BasicBlockType::$name
      }

      fn run(&self, _weights: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
        assert!(inputs.len() == 1);
        vec![inputs[0].map(|x| {
          let mut x = util::fr_to_int(*x) as f32;
          x /= (1 << self.input_SF) as f32;
          x = $operation(x);
          x *= (1 << self.output_SF) as f32;
          Fr::from(x.round() as i32)
        })]
      }
    }
  };
}

make_basic_block!(Exp, ExpBasicBlock, { |x: f32| x.exp() });

make_basic_block!(Log, LogBasicBlock, { |x: f32| x.ln() });

make_basic_block!(ReLU, ReLUBasicBlock, {
  |x: f32| {
    if x < 0f32 {
      0f32
    } else {
      x
    }
  }
});

make_basic_block!(Sqrt, SqrtBasicBlock, { |x: f32| x.sqrt() });

make_basic_block!(ChangeSF, ChangeSFBasicBlock, { |x: f32| x });
