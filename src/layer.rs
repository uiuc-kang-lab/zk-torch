use crate::graph::Graph;
pub use and::AndLayer;
pub use arithmetic::{AddLayer, SubLayer};
use ark_bn254::Fr;
pub use cast::CastLayer;
pub use clip::ClipLayer;
pub use concat::ConcatLayer;
pub use constantofshape::ConstOfShapeLayer;
pub use conv::{ConvLayer, ConvTransposeLayer};
pub use div::{DivLayer, ModLayer};
pub use einsum::EinsumLayer;
pub use equal::EqualLayer;
pub use expand::ExpandLayer;
pub use flatten::FlattenLayer;
pub use gather::GatherLayer;
pub use gathernd::GatherNDLayer;
pub use gemm::GemmLayer;
pub use less::LessLayer;
pub use lstm::LSTMLayer;
pub use matmul::MatMulLayer;
pub use max::{MaxLayer, MinLayer};
pub use mul::MulLayer;
use ndarray::ArrayD;
pub use neg::NegLayer;
pub use nonlinear::*;
pub use norm::{BatchNormLayer, InstanceNormLayer};
pub use not::NotLayer;
pub use pow::PowLayer;
pub use r#where::WhereLayer;
pub use range::RangeLayer;
pub use reducemean::ReduceMeanLayer;
pub use reshape::ReshapeLayer;
pub use resize::ResizeLayer;
pub use scatternd::ScatterNDLayer;
pub use shape::ShapeLayer;
pub use slice::SliceLayer;
pub use softmax::SoftmaxLayer;
pub use split::SplitLayer;
pub use squeeze::{SqueezeLayer, UnsqueezeLayer};
pub use tile::TileLayer;
pub use topk::{ArgMaxLayer, TopKLayer};
use tract_onnx::{pb::AttributeProto, prelude::DatumType};
pub use transpose::TransposeLayer;
pub use xor::XorLayer;

pub mod and;
pub mod arithmetic;
pub mod cast;
pub mod clip;
pub mod concat;
pub mod constantofshape;
pub mod conv;
pub mod div;
pub mod einsum;
pub mod equal;
pub mod expand;
pub mod flatten;
pub mod gather;
pub mod gathernd;
pub mod gemm;
pub mod less;
pub mod lstm;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod neg;
pub mod nonlinear;
pub mod norm;
pub mod not;
pub mod pool;
pub mod pow;
pub mod range;
pub mod reducemean;
pub mod reshape;
pub mod resize;
pub mod scatternd;
pub mod shape;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod squeeze;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod r#where;
pub mod xor;

// Most output types will only depend on an input type but for e.g., Range layer depends on the type of the constants
pub trait Layer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>);
}
