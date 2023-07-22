use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::IxDyn;

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::{layer::{AssignedTensor, CellRc, GadgetConsumer}, dag::{TensorAssignedOrUnassigned, VectorEngine}},};

use super::super::layer::{Layer, LayerConfig};

pub struct PermuteChip {}

impl<F: PrimeField> Layer<F> for PermuteChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let params = &layer_config
      .layer_params
      .iter()
      .map(|x| *x as usize)
      .collect::<Vec<_>>()[..];

    assert!(inp.ndim() == params.len());

    let out = inp.clone();
    let out = out.permuted_axes(IxDyn(params));
    Ok(vec![out])
  }

  fn num_rows(&self, _layer_config: &LayerConfig, _num_cols: i64) -> i64 {
    0
  }
}

impl GadgetConsumer for PermuteChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
