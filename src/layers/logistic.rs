use std::{collections::HashMap, rc::Rc, vec};
use super::dag::{TensorAssignedOrUnassigned, VectorEngine};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  nonlinear::logistic::LogisticGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct LogisticChip {}

impl<F: PrimeField> Layer<F> for LogisticChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
    _vector_engine: &mut VectorEngine<F>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let inp_vec = inp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let zero = constants.get(&0).unwrap().as_ref();

    let logistic_chip = LogisticGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![zero];
    let out = logistic_chip.forward(
      layouter.namespace(|| "logistic chip"),
      &vec_inps,
      &constants,
    )?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let inp_size: usize = layer_config.inp_shapes[0].iter().product();
    let num_inps_per_row = num_cols / 2;
    inp_size.div_ceil(num_inps_per_row as usize) as i64
  }
}

impl GadgetConsumer for LogisticChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::Logistic, GadgetType::InputLookup]
  }
}
