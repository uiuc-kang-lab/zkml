use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

const NUM_COLS_PER_OP: usize = 5;

pub struct MulDivClampChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> MulDivClampChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_map(min_val: i64, num_rows: i64) -> HashMap<i64, i64> {
    let mut map = HashMap::new();
    for i in 0..num_rows {
      let shifted = i + min_val;
      let val = shifted.clamp(0, 255);
      map.insert(i as i64, val);
    }
    map
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let sf = Expression::Constant(F::from(gadget_config.scale_factor));
    let two = Expression::Constant(F::from(2));
    let columns = gadget_config.columns;

    let mut tables = gadget_config.tables;
    let div_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];
    let clamp_lookup = meta.lookup_table_column();

    meta.create_gate("mul_div", |meta| {
      let s = meta.query_selector(selector);

      let mut constraints = vec![];
      for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
        let offset = op_idx * NUM_COLS_PER_OP;
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let mul = meta.query_advice(columns[offset + 1], Rotation::cur());
        let div_res = meta.query_advice(columns[offset + 2], Rotation::cur());
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        let inp_mul = inp.clone() * mul.clone();
        // (2 * inp_mul + sf) = 2 * sf * div + mod
        let lhs = two.clone() * inp_mul.clone() + sf.clone();
        let rhs = two.clone() * sf.clone() * div_res.clone() + mod_res.clone();
        constraints.push((lhs - rhs) * s.clone());
      }

      constraints
    });

    for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
      meta.lookup("mul_div div lookup", |meta| {
        let s = meta.query_selector(selector);
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        // Constrains that mod \in [0, 2 * sf)
        vec![(s.clone() * (two.clone() * sf.clone() - mod_res), div_lookup)]
      });

      meta.lookup("mul_div clamp", |meta| {
        let s = meta.query_selector(selector);
        let div = meta.query_advice(columns[offset + 2], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 4], Rotation::cur());
        let div_outp_min_val = gadget_config.div_outp_min_val;
        let div_outp_min_val = Expression::Constant(F::from((-div_outp_min_val) as u64));

        // Constrains that output \in [0, 255)
        vec![
          (s.clone() * (div + div_outp_min_val), div_lookup),
          (s.clone() * outp, clamp_lookup),
        ]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::MulDivClamp, vec![selector]);

    tables.insert(GadgetType::MulDivClamp, vec![clamp_lookup]);

    let mut maps = gadget_config.maps;
    let clamp_map = Self::get_map(gadget_config.min_val, gadget_config.num_rows as i64);
    maps.insert(GadgetType::MulDivClamp, vec![clamp_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for MulDivClampChip<F> {
  fn name(&self) -> String {
    "MulDivClamp".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    NUM_COLS_PER_OP
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
    let map = &self.config.maps[&GadgetType::MulDivClamp][0];

    let clamp_lookup = self.config.tables[&GadgetType::MulDivClamp][0];

    layouter
      .assign_table(
        || "clamp lookup",
        |mut table| {
          for i in 0..self.config.num_rows {
            let i = i as i64;
            let val = map.get(&i).unwrap();
            table
              .assign_cell(
                || "clamp lookup",
                clamp_lookup,
                i as usize,
                || Value::known(F::from(*val as u64)),
              )
              .unwrap();
          }
          Ok(())
        },
      )
      .unwrap();

    Ok(())
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    _single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let div_val = self.config.scale_factor as i64;

    let div_outp_min_val_i64 = self.config.div_outp_min_val;

    let div_inp_min_val_pos_i64 = -self.config.shift_min_val;
    let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

    let inp = &vec_inputs[0];
    let mul = &vec_inputs[1];
    assert_eq!(inp.len(), mul.len());
    assert_eq!(inp.len() % self.num_inputs_per_row(), 0);

    let clamp_map = &self.config.maps.get(&GadgetType::MulDivClamp).unwrap()[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::MulDivClamp).unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let mut outp_cells = vec![];
    for (i, (inp, mul)) in inp.iter().zip(mul.iter()).enumerate() {
      let offset = i * NUM_COLS_PER_OP;

      let inp_f = inp.value().map(|x: &F| x.to_owned());
      let mul_f = mul.value().map(|x: &F| x.to_owned());

      let mul_res_f = inp_f * mul_f;

      let div_mod_res = mul_res_f.map(|x: F| {
        let x_pos = x + div_inp_min_val_pos;
        let mul_res = convert_to_u64(&x_pos) as i64;
        let div_inp = 2 * mul_res + div_val;
        let div_res = div_inp / (2 * div_val) - div_inp_min_val_pos_i64 / div_val;
        let mod_res = div_inp % (2 * div_val);
        (div_res, mod_res)
      });

      let div_res = div_mod_res.map(|x: (i64, i64)| x.0);
      let mod_res = div_mod_res.map(|x: (i64, i64)| x.1);

      let outp = div_res.map(|x: i64| {
        let mut x_pos = x - div_outp_min_val_i64;
        if !clamp_map.contains_key(&(x_pos)) {
          println!("x: {}, x_pos: {}", x, x_pos);
          x_pos = 0;
        }
        let outp_val = clamp_map.get(&(x_pos)).unwrap();
        F::from(*outp_val as u64)
      });

      // Assign inp, mul
      inp
        .copy_advice(|| "", region, self.config.columns[offset + 0], row_offset)
        .unwrap();
      mul
        .copy_advice(|| "", region, self.config.columns[offset + 1], row_offset)
        .unwrap();

      // Assign div_res, mod_res
      let _div_res_cell = region
        .assign_advice(
          || "div_res",
          self.config.columns[offset + 2],
          row_offset,
          || {
            div_res.map(|x: i64| {
              F::from((x - div_outp_min_val_i64) as u64) - F::from(-div_outp_min_val_i64 as u64)
            })
          },
        )
        .unwrap();
      let _mod_res_cell = region
        .assign_advice(
          || "mod_res",
          self.config.columns[offset + 3],
          row_offset,
          || mod_res.map(|x: i64| F::from(x as u64)),
        )
        .unwrap();

      let outp_cell = region
        .assign_advice(
          || "outp",
          self.config.columns[offset + 4],
          row_offset,
          || outp.map(|x: F| x.to_owned()),
        )
        .unwrap();

      outp_cells.push(outp_cell);
    }

    Ok(outp_cells)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut inps = vec_inputs[0].clone();
    let mut biases = vec_inputs[1].clone();
    let initial_len = inps.len();

    // Needed to pad
    let default = biases[0].clone();
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(&default);
      biases.push(&default);
    }

    let res = self
      .op_aligned_rows(
        layouter.namespace(|| "mul_div_clamp"),
        &vec![inps, biases],
        single_inputs,
      )
      .unwrap();
    Ok(res[0..initial_len].to_vec())
  }
}
