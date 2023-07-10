use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type DotProductBiasConfig = GadgetConfig;

pub struct DotProductBiasChip<F: PrimeField> {
  config: Rc<DotProductBiasConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> DotProductBiasChip<F> {
  pub fn construct(config: Rc<DotProductBiasConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_input_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    let num_inputs = (config.columns.len() - 2) / 2;
    config.columns[0..num_inputs].to_vec()
  }

  pub fn get_weight_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    let num_inputs = (config.columns.len() - 2) / 2;
    config.columns[num_inputs..num_inputs * 2].to_vec()
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = &gadget_config.columns;

    meta.create_gate("dot product bias gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_inp = DotProductBiasChip::<F>::get_input_columns(&gadget_config)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let gate_weights = DotProductBiasChip::<F>::get_weight_columns(&gadget_config)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let bias = meta.query_advice(columns[columns.len() - 2], Rotation::cur());
      let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

      let res = gate_inp
        .iter()
        .zip(gate_weights)
        .map(|(a, b)| a.clone() * b.clone())
        .fold(Expression::Constant(F::ZERO), |a, b| a + b);
      let res = res + bias;

      vec![s * (res - gate_output)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::DotProductBias, vec![selector]);

    GadgetConfig {
      columns: gadget_config.columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for DotProductBiasChip<F> {
  fn name(&self) -> String {
    "dot product bias".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns.len() - 2) / 2
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  // The caller is expected to pad the inputs
  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);

    let inp = &vec_inputs[0];
    let weights = &vec_inputs[1];
    assert_eq!(inp.len(), weights.len());
    assert_eq!(inp.len(), self.num_inputs_per_row());

    let zero = &single_inputs[0];
    let bias = &single_inputs[1];

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::DotProductBias)
        .unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let inp_cols = DotProductBiasChip::<F>::get_input_columns(&self.config);
    inp
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    let weight_cols = DotProductBiasChip::<F>::get_weight_columns(&self.config);
    weights
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, weight_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    bias.copy_advice(
      || "",
      region,
      self.config.columns[self.config.columns.len() - 2],
      row_offset,
    )?;

    // All columns need to be assigned
    if self.config.columns.len() % 2 == 1 {
      zero
        .copy_advice(
          || "",
          region,
          self.config.columns[self.config.columns.len() - 3],
          row_offset,
        )
        .unwrap();
    }

    let e = inp
      .iter()
      .zip(weights.iter())
      .map(|(a, b)| a.value().map(|x: &F| *x) * b.value())
      .reduce(|a, b| a + b)
      .unwrap();
    let e = e + bias.value().map(|x: &F| *x);

    let res = region
      .assign_advice(
        || "",
        self.config.columns[self.config.columns.len() - 1],
        row_offset,
        || e,
      )
      .unwrap();

    Ok(vec![res])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);
    assert!(single_inputs.len() <= 2);
    let zero = single_inputs[0];
    let bias = if single_inputs.len() == 2 {
      single_inputs[1]
    } else {
      single_inputs[0]
    };

    let mut inputs = vec_inputs[0].clone();
    let mut weights = vec_inputs[1].clone();
    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(&zero);
      weights.push(&zero);
    }

    let output = layouter
      .assign_region(
        || "dot prod bias rows",
        |mut region| {
          let mut cur_bias = bias.clone();
          for i in 0..inputs.len() / self.num_inputs_per_row() {
            let inp =
              inputs[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
            let weights =
              weights[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
            cur_bias = self
              .op_row_region(&mut region, i, &vec![inp, weights], &vec![zero, &cur_bias])
              .unwrap()[0]
              .clone();
          }
          Ok(cur_bias)
        },
      )
      .unwrap();

    Ok(vec![output])
  }
}
