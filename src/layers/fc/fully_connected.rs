use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, Error},
};
use ndarray::{Array, ArrayView, IxDyn};

use crate::{
  gadgets::gadget::GadgetConfig,
  utils::helpers::RAND_START_IDX,
};

use super::{
  super::layer::{ActivationType, AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig},
  fc_dp_vdiv_lookup::FCDPVarDivLookupChip,
  fc_dp_vdiv_relud::FCDPVarDivReludChip,
  fc_rlc_vdiv_lookup::FCRLCVarDivLookupChip,
  fc_rlc_vdiv_relud::FCRLCVarDivReludChip,
  fc_dpb_vdiv_lookup::FCDPBiasVarDivLookupChip,
  fc_dpb_vdiv_relud::FCDPBiasVarDivReludChip,
};

#[derive(Clone)]
pub struct FullyConnectedConfig {
  pub normalize: bool, // Should be true
}

impl FullyConnectedConfig {
  pub fn construct(normalize: bool) -> Self {
    Self { normalize }
  }
}

pub struct FullyConnectedChip<F: PrimeField> {
  pub _marker: PhantomData<F>,
  pub config: FullyConnectedConfig,
}

impl<F: PrimeField> FullyConnectedChip<F> {
  pub fn compute_mm(
    // input: &AssignedTensor<F>,
    input: &ArrayView<CellRc<F>, IxDyn>,
    weight: &AssignedTensor<F>,
  ) -> Array<Value<F>, IxDyn> {
    assert_eq!(input.ndim(), 2);
    assert_eq!(weight.ndim(), 2);
    assert_eq!(input.shape()[1], weight.shape()[0]);

    let mut outp = vec![];
    for i in 0..input.shape()[0] {
      for j in 0..weight.shape()[1] {
        let mut sum = input[[i, 0]].value().map(|x: &F| *x) * weight[[0, j]].value();
        for k in 1..input.shape()[1] {
          sum = sum + input[[i, k]].value().map(|x: &F| *x) * weight[[k, j]].value();
        }
        outp.push(sum);
      }
    }

    let out_shape = [input.shape()[0], weight.shape()[1]];
    Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap()
  }

  pub fn assign_array(
    columns: &Vec<Column<Advice>>,
    region: &mut Region<F>,
    array: &Array<Value<F>, IxDyn>,
  ) -> Result<Array<AssignedCell<F, F>, IxDyn>, Error> {
    assert_eq!(array.ndim(), 2);

    let mut outp = vec![];
    for (idx, val) in array.iter().enumerate() {
      let row_idx = idx / columns.len();
      let col_idx = idx % columns.len();
      let cell = region
        .assign_advice(|| "assign array", columns[col_idx], row_idx, || *val)
        .unwrap();
      outp.push(cell);
    }

    let out_shape = [array.shape()[0], array.shape()[1]];
    Ok(Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap())
  }

  pub fn random_vector(
    constants: &HashMap<i64, CellRc<F>>,
    size: usize,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let mut outp = vec![];
    for idx in 0..size {
      let idx = RAND_START_IDX + (idx as i64);
      if !constants.contains_key(&idx) {
        println!("Random vector is too small: {:?}", size);
      }
      let cell = constants.get(&idx).unwrap().clone();
      outp.push(cell);
    }

    Ok(outp)
  }

  pub fn get_activation(layer_params: &Vec<i64>) -> ActivationType {
    let activation = layer_params[0];
    match activation {
      0 => ActivationType::None,
      1 => ActivationType::Relu,
      _ => panic!("Unsupported activation type for fully connected"),
    }
  }
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert!(tensors.len() <= 3);

    let implementation = layer_config.implementation_idx;
    match implementation {
      0 => {
        let chip = FCRLCVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      1 => {
        let chip = FCDPVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      2 => {
        let chip = FCDPBiasVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      3 => {
        let chip = FCRLCVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      4 => {
        let chip = FCDPVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      5 => {
        let chip = FCDPBiasVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.forward(layouter, tensors, constants, gadget_config, layer_config)
      }
      _ => panic!("Unsupported implementation"),
    }
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let implementation = layer_config.implementation_idx;
    match implementation {
      0 => {
        let chip = FCRLCVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      1 => {
        let chip = FCDPVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      2 => {
        let chip = FCDPBiasVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      3 => {
        let chip = FCRLCVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      4 => {
        let chip = FCDPVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      5 => {
        let chip = FCDPBiasVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.num_rows(layer_config, num_cols)
      }
      _ => panic!("Unsupported implementation"),
    }
  }
}

impl<F: PrimeField> GadgetConsumer for FullyConnectedChip<F> {
  fn used_gadgets(&self, layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    let implementation = layer_config.implementation_idx;

    match implementation {
      0 => {
        let chip = FCRLCVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      1 => {
        let chip = FCDPVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      2 => {
        let chip = FCDPBiasVarDivLookupChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      3 => {
        let chip = FCRLCVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      4 => {
        let chip = FCDPVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      5 => {
        let chip = FCDPBiasVarDivReludChip::<F> {
          config: self.config.clone(),
          _marker: PhantomData,
        };
        chip.used_gadgets(layer_config)
      }
      _ => panic!("Unsupported implementation"),
    }
  }
}
