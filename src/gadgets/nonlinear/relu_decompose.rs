use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Region},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct ReluDecomposeChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> ReluDecomposeChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;
    //let k = gadget_config.k;
    let k = columns.len() - 2;
    let k_shift = 1 << (k - 1);

    // Decomposition | input | output
    meta.create_gate("relu decompose", |meta| {
      let s = meta.query_selector(selector);
      let k_shift = Expression::Constant(F::from(k_shift));
      let decomp = columns[0..k]
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();

      // input = decomp[0] + 2 * decomp[1] + ... + 2^{k-1} * decomp[k-1]
      let input_decomp = decomp
        .iter()
        .enumerate()
        .fold(Expression::Constant(F::ZERO), |a, (i, b)| {
          a + Expression::Constant(F::from(1 << i)) * b.clone()
        });
      let input = meta.query_advice(columns[k], Rotation::cur());
      let output = meta.query_advice(columns[k + 1], Rotation::cur());

      vec![
        s.clone() * ((input.clone() + k_shift.clone()) - input_decomp),
        s * ((input - output) * decomp[k - 1].clone()),
      ]
    });

    // Constrain that all of the decomposition bits are in {0, 1}
    let inp_lookup = gadget_config.tables.get(&GadgetType::InputLookup).unwrap()[0];
    for i in 0..k {
      meta.lookup(format!("relu decompose {i}"), |meta| {
        let s = meta.query_selector(selector);
        let col = meta.query_advice(columns[i], Rotation::cur());
        let one = Expression::Constant(F::from(1));
        vec![(s * (one - col), inp_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::ReluDecompose, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for ReluDecomposeChip<F> {
  fn name(&self) -> String {
    "relu decompose".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    //self.config.k + 2
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    1
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    _single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::ReluDecompose)
        .unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let input = vec_inputs[0][0];
    let k = self.config.columns.len()-2;
    let k_shift = 1 << (k - 1);

    // Copy input
    input.copy_advice(|| "", region, self.config.columns[k], row_offset)?;

    // Bit decomposition
    let inp_shift_u64 = input
      .value()
      .map(|x| convert_to_u64(&(*x + F::from(k_shift))));
    for i in 0..k {
      region.assign_advice(
        || "",
        self.config.columns[i],
        row_offset,
        || {
          inp_shift_u64.map(|x| {
            let bit = (x >> i) & 1;
            F::from(bit)
          })
        },
      )?;
    }

    // Assign output
    let output = inp_shift_u64.map(|x| {
      let x_i64 = (x as i64) - (k_shift as i64);
      let relu = x_i64.max(0);
      F::from(relu as u64)
    });
    let outp = region.assign_advice(|| "", self.config.columns[k + 1], row_offset, || output)?;

    Ok(vec![outp])
  }

  fn forward(
    &self,
    mut layouter: impl halo2_proofs::circuit::Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let zero = &single_inputs[0];

    let mut inp = vec_inputs[0].clone();
    let initial_len = inp.len();
    while inp.len() % self.num_inputs_per_row() != 0 {
      inp.push(zero);
    }

    let vec_inputs = vec![inp];
    let res = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      single_inputs,
    )?;
    Ok(res[0..initial_len].to_vec())
  }
}
