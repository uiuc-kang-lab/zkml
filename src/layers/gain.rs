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
    mul_div_clamp::MulDivClampChip,
  },
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::{
  arithmetic::Arithmetic,
  layer::{Layer, LayerConfig},
};

#[derive(Clone, Debug)]
pub struct GainChip {}

impl<F: PrimeField> Arithmetic<F> for GainChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mul_div_clamp_chip = MulDivClampChip::<F>::construct(gadget_config.clone());

    let out = mul_div_clamp_chip.forward(
      layouter.namespace(|| "mul div clamp chip"),
      &vec_inputs,
      constants,
    )?;
    Ok(out)
  }
}

impl<F: PrimeField> Layer<F> for GainChip {
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
}

impl GadgetConsumer for GainChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::MulDivClamp, GadgetType::InputLookup]
  }
}
