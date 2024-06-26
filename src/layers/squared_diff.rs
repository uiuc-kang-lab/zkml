use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig, GadgetType},
    squared_diff::SquaredDiffGadgetChip,
    var_div::VarDivRoundChip,
  },
  utils::helpers::broadcast,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SquaredDiffChip {}

impl<F: PrimeField> Layer<F> for SquaredDiffChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 2);
    let inp1 = &tensors[0];
    let inp2 = &tensors[1];
    // Broadcoasting allowed... can't check shapes easily
    let (inp1, inp2) = broadcast(inp1, inp2);

    let zero = constants.get(&0).unwrap().as_ref();

    let sq_diff_chip = SquaredDiffGadgetChip::<F>::construct(gadget_config.clone());
    let inp1_vec = inp1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let inp2_vec = inp2.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let tmp_constants = vec![zero];
    let out = sq_diff_chip.forward(
      layouter.namespace(|| "sq diff chip"),
      &vec_inputs,
      &tmp_constants,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();

    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| x).collect::<Vec<_>>();
    let out = var_div_chip.forward(
      layouter.namespace(|| "sq diff div"),
      &vec![out],
      &single_inputs,
    )?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp1.shape()), out).unwrap();

    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    // Square diff
    let inp_size: usize = layer_config.inp_shapes[0].iter().product();
    let num_inps_per_row = num_cols / 3;
    let mut num_rows = inp_size.div_ceil(num_inps_per_row as usize) as i64;

    // Divide
    let num_div_per_row = (num_cols - 1) / 3;
    num_rows += (inp_size as i64).div_ceil(num_div_per_row);

    num_rows
  }
}

impl GadgetConsumer for SquaredDiffChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::SquaredDiff,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
