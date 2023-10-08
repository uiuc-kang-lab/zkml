use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::halo2curves::ff::PrimeField;
use ndarray::{Array, ArrayView, Axis, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetType},
    nonlinear::relu_decompose::ReluDecomposeChip,
    var_div::VarDivRoundChip,
  },
  layers::{
    fc::fully_connected::FullyConnectedChip,
    layer::{ActivationType, GadgetConsumer, Layer, LayerConfig},
  },
};

use super::fully_connected::FullyConnectedConfig;

// Implement MM with dot products
// Var div for normalization
// Lookup table for activation
pub struct FCDPVarDivReludChip<F: PrimeField> {
  pub _marker: PhantomData<F>,
  pub config: FullyConnectedConfig,
}

impl<F: PrimeField> Layer<F> for FCDPVarDivReludChip<F> {
  fn forward(
    &self,
    mut layouter: impl halo2_proofs::circuit::Layouter<F>,
    tensors: &Vec<crate::layers::layer::AssignedTensor<F>>,
    constants: &std::collections::HashMap<i64, crate::layers::layer::CellRc<F>>,
    gadget_config: std::rc::Rc<crate::gadgets::gadget::GadgetConfig>,
    layer_config: &crate::layers::layer::LayerConfig,
  ) -> Result<Vec<crate::layers::layer::AssignedTensor<F>>, halo2_proofs::plonk::Error> {
    let input = &tensors[0];
    let ndim = input.ndim();
    let input = if ndim == 2 {
      ArrayView::from(input)
    } else {
      input.index_axis(Axis(0), 0)
    };
    let weight = &tensors[1].t().into_owned();
    let zero = constants.get(&0).unwrap().as_ref();

    let dp_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let mut mm_flat = vec![];
    for i in 0..input.shape()[0] {
      for j in 0..weight.shape()[1] {
        let out_cell = dp_chip.forward(
          layouter.namespace(|| format!("dp_{}_{}", i, j)),
          &vec![
            input
              .index_axis(Axis(0), i)
              .iter()
              .map(|x| x.as_ref())
              .collect(),
            weight
              .index_axis(Axis(1), j)
              .iter()
              .map(|x| x.as_ref())
              .collect(),
          ],
          &vec![zero],
        )?;
        mm_flat.push(out_cell[0].clone());
      }
    }

    let shape = vec![input.shape()[0], weight.shape()[1]];
    let final_flat = if self.config.normalize {
      let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
      let mm_flat_ref = mm_flat.iter().map(|x| x).collect();
      let sf = constants
        .get(&(gadget_config.scale_factor as i64))
        .unwrap()
        .as_ref();
      let mm_div = var_div_chip
        .forward(
          layouter.namespace(|| "mm_div"),
          &vec![mm_flat_ref],
          &vec![zero, sf],
        )
        .unwrap();

      let mm_div = if tensors.len() == 3 {
        let bias = tensors[2].broadcast(shape.clone()).unwrap();
        let bias = bias.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        let mm_div = mm_div.iter().collect::<Vec<_>>();
        let adder_chip = AddPairsChip::<F>::construct(gadget_config.clone());
        let mm_bias = adder_chip
          .forward(
            layouter.namespace(|| "mm_bias"),
            &vec![mm_div, bias],
            &vec![zero],
          )
          .unwrap();
        mm_bias
      } else {
        mm_div
      };

      let activation = FullyConnectedChip::<F>::get_activation(&layer_config.layer_params);
      let mm_div = match activation {
        ActivationType::Relu => {
          let relu_chip = ReluDecomposeChip::<F>::construct(gadget_config.clone());
          let mm_div = mm_div.iter().collect::<Vec<_>>();
          let vec_inputs = vec![mm_div];
          relu_chip
            .forward(layouter.namespace(|| "relu_decompose"), &vec_inputs, &vec![zero])
            .unwrap()
        }
        ActivationType::None => mm_div,
        _ => {
          panic!("Unsupported activation type");
        }
      };

      mm_div.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>()
    } else {
      mm_flat.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>()
    };

    let final_result = Array::from_shape_vec(IxDyn(&shape), final_flat).unwrap();

    Ok(vec![final_result])
  }

  fn num_rows(&self, layer_config: &crate::layers::layer::LayerConfig, num_cols: i64) -> i64 {
    let inp_shape = &layer_config.inp_shapes[0];
    let inp_shape = if inp_shape.len() == 2 {
      inp_shape
    } else {
      &inp_shape[1..]
    };
    let weight_shape = &layer_config.inp_shapes[1];
    let out_shape = &layer_config.out_shapes[0];
    assert_eq!(out_shape.len(), 2);
    let out_size = out_shape.iter().product::<usize>() as i64;

    let mut num_rows =
      <FCDPVarDivReludChip<F> as Layer<F>>::num_rows_dot_acc(inp_shape[1] as i64, num_cols)
        * (inp_shape[0] * weight_shape[0]) as i64;

    // Normalization
    if self.config.normalize {
      let num_divs_per_row = (num_cols - 1) / 3;
      let num_rows_for_div = out_size.div_ceil(num_divs_per_row);
      num_rows += num_rows_for_div;

      if layer_config.inp_shapes.len() == 3 {
        let num_adds_per_row = num_cols / 2;
        let num_rows_for_add = out_size.div_ceil(num_adds_per_row);
        num_rows += num_rows_for_add;
      }

      let activation = FullyConnectedChip::<F>::get_activation(&layer_config.layer_params);
      if activation == ActivationType::Relu {
        num_rows += out_size;
      }
    }

    num_rows
  }
}

impl<F: PrimeField> GadgetConsumer for FCDPVarDivReludChip<F> {
  fn used_gadgets(&self, layer_config: &LayerConfig) -> Vec<GadgetType> {
    let activation = FullyConnectedChip::<F>::get_activation(&layer_config.layer_params);
    let mut outp = vec![
      GadgetType::Adder,
      GadgetType::DotProduct,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ];
    if layer_config.inp_shapes.len() >= 3 {
      outp.push(GadgetType::AddPairs);
    }
    match activation {
      ActivationType::Relu => outp.push(GadgetType::ReluDecompose),
      ActivationType::None => (),
      _ => panic!("Unsupported activation type"),
    }
    outp
  }
}
