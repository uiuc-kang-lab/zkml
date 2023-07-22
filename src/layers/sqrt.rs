use std::{collections::HashMap, rc::Rc, vec};
use super::dag::{TensorAssignedOrUnassigned, VectorEngine};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  nonlinear::sqrt::SqrtGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SqrtChip {}

impl<F: PrimeField> Layer<F> for SqrtChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let mut inp_vec = vec![];

    let mask = &layer_config.mask;
    let mut mask_map = HashMap::new();
    for i in 0..mask.len() / 2 {
      mask_map.insert(mask[2 * i], mask[2 * i + 1]);
    }

    let min_val = gadget_config.min_val;
    let min_val = constants.get(&min_val).unwrap().as_ref();
    let max_val = gadget_config.max_val;
    let max_val = constants.get(&max_val).unwrap().as_ref();
    for (i, val) in inp.iter().enumerate() {
      let i = i as i64;
      if mask_map.contains_key(&i) {
        let mask_val = *mask_map.get(&i).unwrap();
        if mask_val == 1 {
          inp_vec.push(max_val);
        } else if mask_val == -1 {
          inp_vec.push(min_val);
        } else {
          panic!();
        }
      } else {
        inp_vec.push(val.as_ref());
      }
    }

    let zero = constants.get(&0).unwrap().as_ref();
    let sqrt_chip = SqrtGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![zero, min_val, max_val];
    let out = sqrt_chip.forward(layouter.namespace(|| "sqrt chip"), &vec_inps, &constants)?;

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

impl GadgetConsumer for SqrtChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::Sqrt, GadgetType::InputLookup]
  }
}
