use crate::basic_block::*;
use crate::graph::*;
use crate::layer::{conv::reshape_permutation, Layer};
use crate::onnx;
use crate::util::{max_padding_partitions, pad_to_pow_of_two};
use ark_bn254::Fr;
use copy_constraint::zero_padding_partition;
use ndarray::{concatenate, indices, ArrayD, Axis, Dim, Dimension, IxDyn};
use std::collections::BTreeMap;
use tract_onnx::pb::AttributeProto;

// Returns the splat needed to pass into MaxProofBasicBlock. This produces a (product of input dims X 2) permutation where the first column corresponds to the input elements and the second column contains cmp_val
fn splat_input(input_shape: &Vec<usize>, cmp_val: Option<IxDyn>) -> ArrayD<Option<IxDyn>> {
  let inp_shape = Dim(IxDyn(input_shape));
  let inp = ArrayD::from_shape_vec(inp_shape.clone(), indices(inp_shape).into_iter().map(|x| Some(x.into_dyn())).collect()).unwrap();
  let inp = inp.into_shape(IxDyn(&[input_shape.iter().product(), 1])).unwrap();
  let inp_pad = pad_to_pow_of_two(&inp, &cmp_val);
  let second_col = ArrayD::from_elem(inp_pad.shape(), cmp_val);
  concatenate(Axis(1), &[inp_pad.view(), second_col.view()]).unwrap()
}

pub struct MaxLayer;
impl Layer for MaxLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    // For now we only support the case when there are two inputs and the second input is a constant of a single element. The single element is compared element-wise with the first input
    if input_shapes.len() == 2 && input_shapes[1].len() == 1 && constants[1].is_some() {
      let constant = constants[1].unwrap().first().unwrap();
      let permutation = splat_input(&input_shapes[0], None);
      let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
      let padding_partitions = max_padding_partitions(&permutation, *constant);
      let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation,
        input_dim: IxDyn(&input_shape_padded),
        padding_partitions,
      }));
      let max = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MaxProofBasicBlock {}),
        N: 1,
      }));
      let reshape_shape = &vec![input_shapes[0].iter().product(), 1];
      let reshape_permutation = reshape_permutation(&reshape_shape, &input_shapes[0]);
      let padding_partitions = zero_padding_partition(&reshape_permutation);
      let reshape_shape_pad: Vec<_> = reshape_shape.iter().map(|i| i.next_power_of_two()).collect();
      let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: reshape_permutation,
        input_dim: IxDyn(&reshape_shape_pad),
        padding_partitions,
      }));

      let cc_output = graph.addNode(cc, vec![(-1, 0)]);
      let max_output = graph.addNode(max, vec![(cc_output, 0)]);
      let cc1_output = graph.addNode(cc1, vec![(max_output, 0)]);
      graph.outputs.push((cc1_output, 0));
    } else {
      panic!("MaxLayer only supports having two inputs where the second input is a constant")
    }
    (graph, vec![input_shapes[0].clone()])
  }
}

pub struct MinLayer;
impl Layer for MinLayer {
  fn graph(input_shapes: &Vec<&Vec<usize>>, _constants: &Vec<Option<&ArrayD<Fr>>>, _attributes: &Vec<&AttributeProto>) -> (Graph, Vec<Vec<usize>>) {
    let mut graph = Graph::new();
    // For now we only support the case when there are two inputs where the first input has ndim > 1 and the second input is a single element. The single element is compared element-wise with the first input. This is its only use case in RetinaNet where the first input has the same dimensions as the Max output and second input comes from Shape -> Gather layers.
    if input_shapes.len() == 2 && input_shapes[1].len() == 1 && input_shapes[0].len() > 1 {
      // Overview:
      // Make another ArrayD the same size as the first input containing the second input so that they can be concatenated for CopyConstraint
      // Splat it into two columns where the first column contains the first input elements and second contains the second input value
      // Perform element-wise max over the negation of all elements, and negate the outputs
      // Reshape output into the first input shape
      let input_shape_padded: Vec<_> = input_shapes[0].iter().map(|i| i.next_power_of_two()).collect();
      let size = input_shape_padded.iter().product();

      let extended_second_input = ArrayD::from_shape_vec(input_shape_padded.clone(), vec![Some(IxDyn(&[0])); size]).unwrap();

      let extend_second_input = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: extended_second_input,
        input_dim: IxDyn(&[1]),
        padding_partitions: BTreeMap::new(),
      }));

      let concat_inputs = graph.addBB(Box::new(ConcatBasicBlock { axis: 0 }));

      let mut concat_shape = input_shape_padded.clone();
      concat_shape[0] = 2 * concat_shape[0];

      // Pick an arbitrary index corresponding to the second input
      let mut copy_idx = vec![input_shape_padded[0]];
      copy_idx.append(&mut vec![0; input_shapes[0].len() - 1]);

      let permutation = splat_input(&input_shapes[0], Some(IxDyn(&copy_idx)));
      let cc = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation,
        input_dim: IxDyn(&concat_shape),
        padding_partitions: BTreeMap::new(),
      }));

      let neg = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(NegBasicBlock {
          input_SF: onnx::SF,
          output_SF: onnx::SF,
        }),
        N: 1,
      }));

      let max = graph.addBB(Box::new(RepeaterBasicBlock {
        basic_block: Box::new(MaxProofBasicBlock {}),
        N: 1,
      }));

      let reshape_shape = &vec![input_shapes[0].iter().product(), 1];
      let reshape_permutation = reshape_permutation(&reshape_shape, &input_shapes[0]);
      let padding_partitions = zero_padding_partition(&reshape_permutation);
      let reshape_shape_pad: Vec<_> = reshape_shape.iter().map(|i| i.next_power_of_two()).collect();
      let cc1 = graph.addBB(Box::new(CopyConstraintBasicBlock {
        permutation: reshape_permutation,
        input_dim: IxDyn(&reshape_shape_pad),
        padding_partitions,
      }));

      let second_input = graph.addNode(extend_second_input, vec![(-2, 0)]);
      let concat_output = graph.addNode(concat_inputs, vec![(-1, 0), (second_input, 0)]);
      let cc_output = graph.addNode(cc, vec![(concat_output, 0)]);
      let neg_output = graph.addNode(neg, vec![(cc_output, 0)]);
      let max_output = graph.addNode(max, vec![(neg_output, 0)]);
      let neg1_output = graph.addNode(neg, vec![(max_output, 0)]);
      let cc1_output = graph.addNode(cc1, vec![(neg1_output, 0)]);
      graph.outputs.push((cc1_output, 0));
    } else {
      panic!("MinLayer only supports having two inputs where the first input has ndim > 1 and the second input is a single element")
    }
    (graph, vec![input_shapes[0].clone()])
  }
}
