use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, Axis, IxDyn};

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::fc::fully_connected::{FullyConnectedChip, FullyConnectedConfig},
};

use super::{layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig}, dag::{TensorAssignedOrUnassigned, VectorEngine}};

pub struct BatchMatMulChip {}

impl<F: PrimeField> Layer<F> for BatchMatMulChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _flex_tensors: &Vec<TensorAssignedOrUnassigned<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp1 = &tensors[0];
    let inp2 = &tensors[1];
    println!("inp1: {:?}", inp1.shape());
    println!("inp2: {:?}", inp2.shape());

    assert_eq!(inp1.ndim(), 3);
    assert_eq!(inp2.ndim(), 3);
    assert_eq!(inp1.shape()[0], inp2.shape()[0]);

    let adj_y = layer_config.layer_params[1] == 1;
    if adj_y {
      assert_eq!(inp1.shape()[2], inp2.shape()[2]);
    } else {
      assert_eq!(inp1.shape()[2], inp2.shape()[1]);
    }

    let out_shape = if adj_y {
      vec![inp1.shape()[0], inp1.shape()[1], inp2.shape()[1]]
    } else {
      vec![inp1.shape()[0], inp1.shape()[1], inp2.shape()[2]]
    };

    let fc_chip = FullyConnectedChip::<F> {
      _marker: PhantomData,
      config: FullyConnectedConfig::construct(true),
    };

    let mut outp: Vec<CellRc<F>> = vec![];
    for i in 0..inp1.shape()[0] {
      let inp1_slice = inp1.index_axis(Axis(0), i).to_owned();
      // Due to tensorflow BS, transpose the "weights"
      let inp2_slice = if adj_y {
        inp2.index_axis(Axis(0), i).to_owned()
      } else {
        inp2.index_axis(Axis(0), i).t().to_owned()
      };
      println!("inp1_slice: {:?}", inp1_slice.shape());
      println!("inp2_slice: {:?}", inp2_slice.shape());
      // Batch MM doesn't have a fused activation, so insert it here
      // TODO: consider putting this in the converter?
      let tmp_config = LayerConfig {
        layer_params: vec![0],
        ..layer_config.clone()
      };
      let outp_slice = fc_chip.forward(
        layouter.namespace(|| ""),
        &vec![inp1_slice, inp2_slice],
        &vec![],
        constants,
        gadget_config.clone(),
        &tmp_config,
        vector_engine,
      )?;
      outp.extend(outp_slice[0].iter().map(|x| x.clone()).collect::<Vec<_>>());
    }

    let outp = Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap();
    Ok(vec![outp])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let adj_y = layer_config.layer_params[1] == 1;
    let (inp1_shape, inp2_shape) = if adj_y {
      (
        layer_config.inp_shapes[0].clone(),
        layer_config.inp_shapes[1].clone(),
      )
    } else {
      (
        layer_config.inp_shapes[0].clone(),
        vec![
          layer_config.inp_shapes[1][0],
          layer_config.inp_shapes[1][2],
          layer_config.inp_shapes[1][1],
        ],
      )
    };

    let one_mm_num_rows = {
      let inp1_one_shape = vec![inp1_shape[1], inp1_shape[2]];
      let inp2_one_shape = vec![inp2_shape[1], inp2_shape[2]];
      let tmp_config = LayerConfig {
        layer_params: vec![0],
        inp_shapes: vec![inp1_one_shape, inp2_one_shape],
        ..layer_config.clone()
      };
      let fc_chip = FullyConnectedChip::<F> {
        _marker: PhantomData,
        config: FullyConnectedConfig::construct(true),
      };
      fc_chip.num_rows(&tmp_config, num_cols)
    };

    inp1_shape[0] as i64 * one_mm_num_rows
  }
}

impl GadgetConsumer for BatchMatMulChip {
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Adder,
      GadgetType::DotProduct,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
