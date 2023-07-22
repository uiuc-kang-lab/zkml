use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::max_pool_2d::MaxPool2DChip,
};

use super::{
  averager::Averager,
  layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig}, dag::{TensorAssignedOrUnassigned, VectorEngine},
};

pub struct AvgPool2DChip {}

impl<F: PrimeField> Averager<F> for AvgPool2DChip {
  fn splat(&self, input: &AssignedTensor<F>, layer_config: &LayerConfig) -> Vec<Vec<CellRc<F>>> {
    assert_eq!(input.shape().len(), 4);
    // Don't support batch size > 1 yet
    assert_eq!(input.shape()[0], 1);

    // TODO: refactor this
    MaxPool2DChip::splat(input, layer_config).unwrap()
  }

  fn get_div_val(
    &self,
    mut layouter: impl Layouter<F>,
    _tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
   ) -> Result<AssignedCell<F, F>, Error> {
    // FIXME: this needs to be revealed
    let div = layer_config.layer_params[0] * layer_config.layer_params[1];
    let div = F::from(div as u64);
    let div = layouter
      .assign_region(
        || "avg pool 2d div",
        |mut region| {
          let div = region
            .assign_advice(
              || "avg pool 2d div",
              gadget_config.columns[0],
              0,
              || Value::known(div),
            )
            .unwrap();
          Ok(div)
        },
      )
      .unwrap();

    Ok(div)
  }
}

impl<F: PrimeField> Layer<F> for AvgPool2DChip {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let dived = self
      .avg_forward(layouter, tensors, constants, gadget_config, layer_config)
      .unwrap();

    let inp = &tensors[0];
    // TODO: refactor this
    let out_xy = MaxPool2DChip::<F>::shape(inp.shape(), layer_config);
    let out_shape = vec![1, out_xy.0, out_xy.1, inp.shape()[3]];
    println!("out_shape: {:?}", out_shape);

    let out = Array::from_shape_vec(IxDyn(&out_shape), dived).unwrap();
    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let out_shape = &layer_config.out_shapes[0];

    let num_sums: usize = out_shape.iter().product();
    let num_inps_per_sum = layer_config.layer_params[0] * layer_config.layer_params[1];

    Averager::<F>::get_num_rows(self, num_sums as i64, num_inps_per_sum as i64, num_cols)
  }
}

impl GadgetConsumer for AvgPool2DChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Adder,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
