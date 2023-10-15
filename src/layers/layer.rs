use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::{GadgetConfig, GadgetType};

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum LayerType {
  Add,
  AvgPool2D,
  BatchMatMul,
  Broadcast,
  Concatenation,
  Conv2D,
  Cos,
  DivVar,
  DivFixed,
  FullyConnected,
  Logistic,
  MaskNegInf,
  MaxPool2D,
  Mean,
  Mul,
  #[default]
  Noop,
  Pack,
  Pad,
  Pow,
  Permute,
  Reshape,
  ResizeNN,
  Rotate,
  Rsqrt,
  Sin,
  Slice,
  Softmax,
  Split,
  Sqrt,
  Square,
  SquaredDifference,
  Sub,
  Tanh,
  Transpose,
  Update,
}

// NOTE: This is the same order as the TFLite schema
// Must not be changed
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum ActivationType {
  #[default]
  None,
  Relu,
  ReluN1To1,
  Relu6,
  Tanh,
  SignBit,
}

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
  pub layer_type: LayerType,
  pub layer_params: Vec<i64>, // This is turned into layer specific configurations at runtime
  pub inp_shapes: Vec<Vec<usize>>,
  pub out_shapes: Vec<Vec<usize>>,
  pub mask: Vec<i64>,
  pub implementation_idx: i64,
}

pub type CellRc<F> = Rc<AssignedCell<F, F>>;
pub type AssignedTensor<F> = Array<CellRc<F>, IxDyn>;
// General issue with rust: I'm not sure how to pass named arguments to a trait...
// Currently, the caller must be aware of the order of the tensors and results
pub trait Layer<F: PrimeField> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error>;

  // The layer config has the input and output sizes (hypothetically...)
  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64;

  fn num_rows_reduction(num_elems: i64, num_elem_per_row: i64) -> i64 {
    let mut num_rows = num_elems.div_ceil(num_elem_per_row);
    let mut residual = num_rows;
    while residual > 1 {
      residual = residual.div_ceil(num_elem_per_row);
      num_rows += residual;
    }
    num_rows
  }

  // Number of rows for the dot product with addition accumulator
  fn num_rows_dot_acc(len: i64, num_cols: i64) -> i64 {
    let inps_per_row = (num_cols - 1) / 2;
    let num_rows_for_dot = len.div_ceil(inps_per_row);

    let num_adds_per_row = num_cols - 1;
    let num_rows_for_acc = Self::num_rows_reduction(num_rows_for_dot, num_adds_per_row);

    num_rows_for_dot + num_rows_for_acc
  }

  fn num_rows_dot_bias(len: i64, num_cols: i64) -> i64 {
    let inps_per_row = (num_cols - 2) / 2;
    len.div_ceil(inps_per_row)
  }
}

pub trait GadgetConsumer {
  fn used_gadgets(&self, layer_config: &LayerConfig) -> Vec<GadgetType>;
}
