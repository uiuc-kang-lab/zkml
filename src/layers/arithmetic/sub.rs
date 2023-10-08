use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig, GadgetType},
    sub_pairs::SubPairsChip,
  },
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::{
  super::layer::{Layer, LayerConfig},
  Arithmetic,
};

#[derive(Clone, Debug)]
pub struct SubChip {}

impl<F: PrimeField> Arithmetic<F> for SubChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let sub_pairs_chip = SubPairsChip::<F>::construct(gadget_config);
    let out = sub_pairs_chip.forward(layouter.namespace(|| "sub chip"), &vec_inputs, constants)?;
    Ok(out)
  }
}

impl<F: PrimeField> Layer<F> for SubChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let (out, out_shape) = self.arithmetic_forward(
      layouter.namespace(|| ""),
      tensors,
      constants,
      gadget_config.clone(),
    )?;
    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();

    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let inp_size = <SubChip as Arithmetic<F>>::get_inp_size(layer_config);

    let num_sub_per_row = num_cols / 3;
    (inp_size as i64).div_ceil(num_sub_per_row)
  }
}

impl GadgetConsumer for SubChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::SubPairs]
  }
}
