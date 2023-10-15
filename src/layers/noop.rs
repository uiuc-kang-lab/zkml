use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};

use crate::gadgets::gadget::GadgetConfig;

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

use ndarray::{Array, IxDyn};

pub struct NoopChip {}

impl<F: PrimeField> Layer<F> for NoopChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let ret_idx = layer_config.layer_params[0] as usize;
    let out_shape = layer_config.out_shapes[0].clone();
    let len_out = out_shape.iter().product();
    let zero = constants.get(&0).unwrap();

    let unshaped_tensor = tensors[ret_idx].clone();
    let mut inp_flat = unshaped_tensor.iter().cloned().collect::<Vec<_>>();
    if inp_flat.len() > len_out {
      inp_flat = inp_flat[..len_out].into_iter().cloned().collect::<Vec<_>>();
    } else if inp_flat.len() < len_out {
      inp_flat.resize(len_out, zero.clone()); // This is not correct, fix it later
    } 

    let shaped_tensor = Array::from_shape_vec(IxDyn(&out_shape), inp_flat).unwrap();

    Ok(vec![shaped_tensor])
  }

  fn num_rows(&self, _layer_config: &LayerConfig, _num_cols: i64) -> i64 {
    0
  }
}

impl GadgetConsumer for NoopChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
