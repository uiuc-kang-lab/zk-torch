use super::{BasicBlock, BasicBlockType};
use ark_bn254::Fr;
use ark_std::Zero;
use ndarray::{ArrayD, Axis};

// no prove and verify for squeeze and unsqueeze

// currently, we only support squeeze the first axis
#[derive(Debug)]
pub struct SqueezeBasicBlock;

impl BasicBlock for SqueezeBasicBlock {
  fn block_type(&self) -> Result<BasicBlockType, String> {
    Ok(BasicBlockType::Squeeze)
  }

  fn run(&self, _weights: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // squeeze the input tensor
    assert!(inputs.len() == 1);
    let r = inputs[0].clone();
    let r = r.remove_axis(Axis(0));
    vec![r]
  }
}

// currently, we only support unsqueeze the first axis
#[derive(Debug)]
pub struct UnsqueezeBasicBlock;

impl BasicBlock for UnsqueezeBasicBlock {
  fn block_type(&self) -> Result<BasicBlockType, String> {
    Ok(BasicBlockType::Unsqueeze)
  }

  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    // unsqueeze the input tensor
    assert!(inputs.len() == 1);
    let r = inputs[0].clone();
    let r = r.insert_axis(Axis(0));
    vec![r]
  }
}
