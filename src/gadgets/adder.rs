use std::marker::PhantomData;

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type AdderConfig = GadgetConfig;

pub struct AdderChip<F: FieldExt> {
  config: AdderConfig,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> AdderChip<F> {
  pub fn construct(config: AdderConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    meta.create_gate("adder gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_inp = columns[0..columns.len() - 1]
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

      let res = gate_inp
        .iter()
        .fold(Expression::Constant(F::zero()), |a, b| {
          a.clone() + b.clone()
        });

      vec![s * (res - gate_output.clone())]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Adder, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }

  pub fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() - 1
  }
}

// NOTE: The forward pass of the adder adds _everything_ into one cell
impl<F: FieldExt> Gadget<F> for AdderChip<F> {
  fn name(&self) -> String {
    "adder".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  // The caller is expected to pad the inputs
  fn op_row(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    _single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 1);
    let inp = &vec_inputs[0];
    let selector = self.config.selectors.get(&GadgetType::Adder).unwrap()[0];

    let output_cell = layouter.assign_region(
      || "",
      |mut region| {
        selector.enable(&mut region, 0)?;

        inp
          .iter()
          .enumerate()
          .map(|(i, cell)| cell.copy_advice(|| "", &mut region, self.config.columns[i], 0))
          .collect::<Result<Vec<_>, _>>()?;

        let e = inp.iter().fold(Value::known(F::from(0)), |a, b| {
          a + b.value().map(|x: &F| x.to_owned())
        });
        let res = region.assign_advice(
          || "",
          self.config.columns[self.config.columns.len() - 1],
          0,
          || e,
        )?;
        Ok(res)
      },
    )?;

    Ok(vec![output_cell])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(single_inputs.len(), 1);

    let mut inputs = vec_inputs[0].clone();
    let zero = single_inputs[0].clone();

    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(zero.clone());
    }

    let mut outputs = self.op_aligned_rows(
      layouter.namespace(|| "adder forward"),
      &vec![inputs],
      single_inputs,
    )?;
    while outputs.len() != 1 {
      while outputs.len() % self.num_inputs_per_row() != 0 {
        outputs.push(zero.clone());
      }
      outputs = self.op_aligned_rows(
        layouter.namespace(|| "adder forward"),
        &vec![outputs],
        single_inputs,
      )?;
    }

    Ok(outputs)
  }
}
