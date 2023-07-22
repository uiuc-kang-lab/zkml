use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig, GadgetType},
    mul_pairs::MulPairsChip,
    var_div::VarDivRoundChip,
  },
  layers::{layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig}, dag::TensorAssignedOrUnassigned},
};

use super::Arithmetic;

pub struct DivVarChip {}

// TODO: hack. Used for multiplying by the scale factor
impl<F: PrimeField> Arithmetic<F> for DivVarChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mul_pairs_chip = MulPairsChip::<F>::construct(gadget_config.clone());

    let out = mul_pairs_chip.forward(
      layouter.namespace(|| "mul pairs chip"),
      &vec_inputs,
      constants,
    )?;
    Ok(out)
  }
}

impl<F: PrimeField> Layer<F> for DivVarChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &crate::layers::layer::LayerConfig,
    vector_engine: &mut crate::layers::dag::VectorEngine<F>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 2);
    // TODO: We only support dividing by a single number for now
    assert_eq!(tensors[1].shape().len(), 1);
    assert_eq!(tensors[1].shape()[0], 1);

    let sf = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();

    let sf_tensor = Array::from_shape_vec(IxDyn(&[1]), vec![Rc::new(sf.clone())]).unwrap();

    // out = inp * SF
    let (out, out_shape) = self.arithmetic_forward(
      layouter.namespace(|| ""),
      &vec![tensors[0].clone(), sf_tensor],
      constants,
      gadget_config.clone(),
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = tensors[1].iter().next().unwrap().as_ref();
    let zero = constants.get(&0).unwrap().as_ref();
    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let out = var_div_chip.forward(layouter.namespace(|| "mul div"), &vec![out], &single_inputs)?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();
    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let inp_size = <DivVarChip as Arithmetic<F>>::get_inp_size(layer_config);

    // Multiplication
    let num_mul_per_row = num_cols / 3;
    let mut num_rows = (inp_size as i64).div_ceil(num_mul_per_row);

    // Division by the scale factor
    // FIXME: should be taken from the gadgets...
    let num_div_per_row = (num_cols - 1) / 3;
    num_rows += (inp_size as i64).div_ceil(num_div_per_row);

    num_rows
  }
}

impl GadgetConsumer for DivVarChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::MulPairs,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
