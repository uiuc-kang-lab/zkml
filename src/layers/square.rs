use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  square::SquareGadgetChip,
  var_div::VarDivRoundChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SquareChip {}

impl<F: PrimeField> Layer<F> for SquareChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 1);

    let inp = &tensors[0];
    let zero = constants.get(&0).unwrap().as_ref();

    let square_chip = SquareGadgetChip::<F>::construct(gadget_config.clone());
    let inp_vec = inp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let vec_inputs = vec![inp_vec];
    let single_inps = vec![zero];
    let out = square_chip.forward(
      layouter.namespace(|| "square chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    let single_inps = vec![zero, div];
    let out = out.iter().collect::<Vec<_>>();
    let vec_inputs = vec![out];
    let out = var_div_chip.forward(
      layouter.namespace(|| "var div chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();
    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    // Square
    let inp_size: usize = layer_config.inp_shapes[0].iter().product();
    let num_inps_per_row = num_cols / 2;
    let mut num_rows = inp_size.div_ceil(num_inps_per_row as usize) as i64;

    // Divide
    let num_div_per_row = (num_cols - 1) / 3;
    num_rows += (inp_size as i64).div_ceil(num_div_per_row);

    num_rows
  }
}

impl GadgetConsumer for SquareChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Square,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
