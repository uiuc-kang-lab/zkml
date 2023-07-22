use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, ArrayView, Axis, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    dot_prod::DotProductChip,
    gadget::Gadget,
    gadget::{GadgetConfig, GadgetType},
    nonlinear::relu::ReluChip,
    var_div::VarDivRoundChip,
  },
  layers::{
    fc::fully_connected::FullyConnectedChip,
    layer::{ActivationType, AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig},
  },
};

use super::fully_connected::FullyConnectedConfig;

pub struct FCRLCVarDivLookupChip<F: PrimeField> {
  pub _marker: PhantomData<F>,
  pub config: FullyConnectedConfig,
}

impl<F: PrimeField> Layer<F> for FCRLCVarDivLookupChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let activation = FullyConnectedChip::<F>::get_activation(&layer_config.layer_params);

    let input = &tensors[0];
    let ndim = input.ndim();
    let input = if ndim == 2 {
      ArrayView::from(input)
    } else {
      input.index_axis(Axis(0), 0)
    };
    let weight = &tensors[1].t().into_owned();

    let zero = constants.get(&0).unwrap().as_ref();

    // Compute and assign the result
    let mm_result = layouter
      .assign_region(
        || "compute and assign mm",
        |mut region| {
          let mm_result = FullyConnectedChip::compute_mm(&input, weight);
          let mm_result =
            FullyConnectedChip::assign_array(&gadget_config.columns, &mut region, &mm_result)
              .unwrap();

          Ok(mm_result)
        },
      )
      .unwrap();

    // Generate random vectors
    let r1 = FullyConnectedChip::random_vector(constants, mm_result.shape()[0]).unwrap();
    let r2 = FullyConnectedChip::random_vector(constants, mm_result.shape()[1]).unwrap();

    let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let r1_ref = r1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let r2_ref = r2.iter().map(|x| x.as_ref()).collect::<Vec<_>>();

    // Compute r1 * result
    let mut r1_res = vec![];
    // println!("r1_ref: {:?}", r1_ref.len());
    // println!("r2_ref: {:?}", r2_ref.len());
    // println!("mm_result: {:?}", mm_result.shape());
    for i in 0..mm_result.shape()[1] {
      let tmp = mm_result.index_axis(Axis(1), i);
      let mm_ci = tmp.iter().collect::<Vec<_>>();
      let r1_res_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("r1_res_{}", i)),
          &vec![mm_ci, r1_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      r1_res.push(r1_res_i[0].clone());
    }

    // Compute r1 * result * r2
    let r1_res_ref = r1_res.iter().collect::<Vec<_>>();
    let r1_res_r2 = dot_prod_chip
      .forward(
        layouter.namespace(|| "r1_res_r2"),
        &vec![r1_res_ref, r2_ref.clone()],
        &vec![zero],
      )
      .unwrap();
    let r1_res_r2 = r1_res_r2[0].clone();
    // println!("r1_res_r2: {:?}", r1_res_r2);

    // Compute r1 * input
    let mut r1_input = vec![];
    // println!("input: {:?}", input.shape());
    // println!("r1_ref: {:?}", r1_ref.len());
    for i in 0..input.shape()[1] {
      let tmp = input.index_axis(Axis(1), i);
      let input_ci = tmp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let r1_input_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("r1_input_{}", i)),
          &vec![input_ci, r1_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      r1_input.push(r1_input_i[0].clone());
    }

    // Compute weight * r2
    let mut weight_r2 = vec![];
    for i in 0..weight.shape()[0] {
      let tmp = weight.index_axis(Axis(0), i);
      let weight_ci = tmp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let weight_r2_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("weight_r2_{}", i)),
          &vec![weight_ci, r2_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      weight_r2.push(weight_r2_i[0].clone());
    }

    // Compute (r1 * input) * (weight * r2)
    let r1_input_ref = r1_input.iter().collect::<Vec<_>>();
    let weight_r2_ref = weight_r2.iter().collect::<Vec<_>>();
    let r1_inp_weight_r2 = dot_prod_chip
      .forward(
        layouter.namespace(|| "r1_inp_weight_r2"),
        &vec![r1_input_ref, weight_r2_ref],
        &vec![zero],
      )
      .unwrap();

    let r1_inp_weight_r2 = r1_inp_weight_r2[0].clone();
    // println!("r1_inp_weight_r2: {:?}", r1_inp_weight_r2);

    layouter
      .assign_region(
        || "fc equality check",
        |mut region| {
          let t1 = r1_res_r2
            .copy_advice(|| "", &mut region, gadget_config.columns[0], 0)
            .unwrap();
          let t2 = r1_inp_weight_r2
            .copy_advice(|| "", &mut region, gadget_config.columns[0], 1)
            .unwrap();

          region.constrain_equal(t1.cell(), t2.cell()).unwrap();

          Ok(())
        },
      )
      .unwrap();

    let shape = [mm_result.shape()[0], mm_result.shape()[1]];
    let final_result_flat = if self.config.normalize {
      let mm_flat = mm_result.iter().collect::<Vec<_>>();
      let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
      let sf = constants
        .get(&(gadget_config.scale_factor as i64))
        .unwrap()
        .as_ref();
      let mm_div = var_div_chip
        .forward(
          layouter.namespace(|| "mm_div"),
          &vec![mm_flat],
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

      let mm_div = if activation == ActivationType::Relu {
        let relu_chip = ReluChip::<F>::construct(gadget_config.clone());
        let mm_div = mm_div.iter().collect::<Vec<_>>();
        let vec_inputs = vec![mm_div];
        relu_chip
          .forward(layouter.namespace(|| "relu"), &vec_inputs, &vec![zero])
          .unwrap()
      } else if activation == ActivationType::None {
        mm_div
      } else {
        panic!("Unsupported activation type");
      };

      mm_div.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>()
    } else {
      mm_result
        .into_iter()
        .map(|x| Rc::new(x))
        .collect::<Vec<_>>()
    };
    let final_result = Array::from_shape_vec(IxDyn(&shape), final_result_flat).unwrap();

    Ok(vec![final_result])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    // Assign the result
    let out_shape = &layer_config.out_shapes[0];
    println!("out_shape: {:?}", out_shape);

    // assert_eq!(out_shape.len(), 2);
    
    let out_size = out_shape.iter().product::<usize>() as i64;
    let mut num_rows = out_size.div_ceil(num_cols);

    // // r1 * result
    num_rows +=
      (<FCRLCVarDivLookupChip<F> as Layer<F>>::num_rows_dot_acc(out_shape[0] as i64, num_cols)
        * (out_shape[1] as i64)) / 2;
    // r1 * result * r2
    num_rows +=
      (<FCRLCVarDivLookupChip<F> as Layer<F>>::num_rows_dot_acc(out_shape[1] as i64, num_cols)) / 2;

    // r1 * input
    let inp_shape = &layer_config.inp_shapes[0];
    let inp_shape = if inp_shape.len() == 2 {
      inp_shape
    } else {
      &inp_shape[1..]
    };
    num_rows +=
      <FCRLCVarDivLookupChip<F> as Layer<F>>::num_rows_dot_acc(inp_shape[0] as i64, num_cols)
        * (inp_shape[1] as i64) / 2;

    // let weight_shape = &layer_config.inp_shapes[1];
    // num_rows +=
    //   <FCRLCVarDivLookupChip<F> as Layer<F>>::num_rows_dot_acc(weight_shape[0] as i64, num_cols)
    //     * (weight_shape[1] as i64);

    // (r1 * input) * (weight * r2)
    num_rows +=
      <FCRLCVarDivLookupChip<F> as Layer<F>>::num_rows_dot_acc(inp_shape[1] as i64, num_cols) / 2;

      // println!("{:?}"
    // Normalization
    if self.config.normalize {
      let num_divs_per_row = (num_cols - 1) / 4;
      let num_rows_for_div = out_size.div_ceil(num_divs_per_row);
      num_rows += num_rows_for_div;

      if layer_config.inp_shapes.len() == 3 {
        let num_adds_per_row = num_cols / 2;
        let num_rows_for_add = out_size.div_ceil(num_adds_per_row);
        num_rows += num_rows_for_add;
      }

      let activation = FullyConnectedChip::<F>::get_activation(&layer_config.layer_params);
      if activation == ActivationType::Relu {
        let num_relus_per_row = num_cols / 2;
        let num_rows_for_relu = out_size.div_ceil(num_relus_per_row);
        num_rows += num_rows_for_relu;
      }
    }

    num_rows
  }
}

impl<F: PrimeField> GadgetConsumer for FCRLCVarDivLookupChip<F> {
  fn used_gadgets(&self, layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
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
      ActivationType::Relu => outp.push(GadgetType::Relu),
      ActivationType::None => (),
      _ => panic!("Unsupported activation type"),
    }
    outp
  }
}
