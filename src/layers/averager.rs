use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};

use crate::gadgets::gadget::Gadget;
use crate::gadgets::{adder::AdderChip, gadget::GadgetConfig, var_div::VarDivRoundChip};

use super::{
  arithmetic::add::AddChip,
  layer::{AssignedTensor, CellRc, Layer, LayerConfig},
};

pub trait Averager<F: PrimeField> {
  fn splat(&self, input: &AssignedTensor<F>, layer_config: &LayerConfig) -> Vec<Vec<CellRc<F>>>;

  fn get_div_val(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<AssignedCell<F, F>, Error>;

  fn get_num_rows(&self, num_sums: i64, num_inps_per_sum: i64, num_cols: i64) -> i64 {
    // Number of rows from the sum
    let num_adds_per_row = num_cols - 1;
    let num_rows_per_sum =
      <AddChip as Layer<F>>::num_rows_reduction(num_inps_per_sum, num_adds_per_row);
    let mut num_rows = num_sums * num_rows_per_sum;
    
    // For the revealed avg pool 2d div
    num_rows += 1;

    // Number of rows from the division
    let num_divs_per_row = (num_cols - 1) / 3;
    num_rows += num_sums.div_ceil(num_divs_per_row);

    num_rows
  }

  fn avg_forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<CellRc<F>>, Error> {
    // Due to Mean BS
    // assert_eq!(tensors.len(), 1);
    let zero = constants.get(&0).unwrap().as_ref();

    let inp = &tensors[0];
    let splat_inp = self.splat(inp, layer_config);

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let single_inputs = vec![zero];
    let mut added = vec![];
    for i in 0..splat_inp.len() {
      let tmp = splat_inp[i].iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let tmp = adder_chip.forward(
        layouter.namespace(|| format!("average {}", i)),
        &vec![tmp],
        &single_inputs,
      )?;
      added.push(tmp[0].clone());
    }

    let div = self.get_div_val(
      layouter.namespace(|| "average div"),
      tensors,
      gadget_config.clone(),
      layer_config,
    )?;
    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let single_inputs = vec![zero, &div];
    let added = added.iter().map(|x| x).collect::<Vec<_>>();
    let dived = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![added],
      &single_inputs,
    )?;
    let dived = dived.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();

    Ok(dived)
  }
}
