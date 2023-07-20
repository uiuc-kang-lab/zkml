use std::{collections::{HashMap, BTreeMap}, fs::File, io::BufWriter, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::{Layouter, Value}, halo2curves::ff::PrimeField, plonk::{Error, MatrixConfig, Advice, Column, Assigned}};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::{convert_to_u64, GadgetConfig},
  layers::{
    arithmetic::{add::AddChip, div_var::DivVarChip, mul::MulChip, sub::SubChip},
    batch_mat_mul::BatchMatMulChip,
    div_fixed::DivFixedChip,
    fc::fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    logistic::LogisticChip,
    max_pool_2d::MaxPool2DChip,
    mean::MeanChip,
    noop::NoopChip,
    pow::PowChip,
    rsqrt::RsqrtChip,
    shape::{
      broadcast::BroadcastChip, concatenation::ConcatenationChip, mask_neg_inf::MaskNegInfChip,
      pack::PackChip, pad::PadChip, permute::PermuteChip, reshape::ReshapeChip,
      resize_nn::ResizeNNChip, rotate::RotateChip, slice::SliceChip, split::SplitChip,
      transpose::TransposeChip,
    },
    softmax::SoftmaxChip,
    sqrt::SqrtChip,
    square::SquareChip,
    squared_diff::SquaredDiffChip,
    tanh::TanhChip,
    update::UpdateChip,
  },
  utils::helpers::print_assigned_arr, model::MatrixLog,
};

use super::{
  avg_pool_2d::AvgPool2DChip,
  conv2d::Conv2DChip,
  layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig, LayerType},
};

#[derive(Clone, Debug, Default)]
pub struct DAGLayerConfig {
  pub ops: Vec<LayerConfig>,
  pub inp_idxes: Vec<Vec<usize>>,
  pub out_idxes: Vec<Vec<usize>>,
  pub final_out_idxes: Vec<usize>,
}

pub struct DAGLayerChip<F: PrimeField + Ord> {
  dag_config: DAGLayerConfig,
  _marker: PhantomData<F>,
}

// You turn each of the matmul assignments into a 'ticket' that we store for later processing
pub struct MatmulAssignment<F: PrimeField + Ord> {
  width: usize,
  height: usize,
  tensor: Vec<Vec<F>>,
  input_vec: Vec<CellRc<F>>,
  output_vec: Vec<CellRc<F>>, 
}

/// The vector 
/// We also want to actually 
pub struct VectorEngine<F: PrimeField + Ord> {
  assignments: HashMap<MatrixConfig, Vec<MatmulAssignment<F>>>,
}

/// The final matrix
pub struct MatrixOutput<F: PrimeField + Ord> {
  /// stacked input vector
  input_vector: Vec<CellRc<F>>,
  /// Accumulated output vector
  output_vector: Vec<CellRc<F>>,
  /// stacked tensor
  tensor: Vec<Vec<F>>,
}

impl<F: PrimeField + Ord> VectorEngine<F> {
  // So right now, for our batchmul thing, we are essentially doing a gate where we compute the frievalded output. Right now, we do a batch size
  // 1. Create the frievalds' ready
  // 
  fn assign_matmul(
    &mut self,
    matrix_log: &MatrixLog,
    input_vec: Vec<CellRc<F>>,
  ) {
    panic!("not implemented");
  }

  pub fn generate_column_assignments() -> Vec<MatrixOutput<F>> {
    panic!("not implemented");
  }
}

// impl<F: PrimeField + Ord> MatrixOutput<F> {
//   // So after we've aggregated all the stuff, we can now finally assign our matrices
//   fn assign_matrix_output(
//     &self,
//     mut layouter: impl Layouter<F>,
//   ) -> Result<(), Error> {
//     layouter.assign_region("region", ||)
//     Ok(())
//   }
// }

pub enum TensorAssignedOrUnassigned<F: PrimeField + Ord> {
  Unassigned(Array<F, IxDyn>),
  Assigned(AssignedTensor<F>)
}

impl<F: PrimeField + Ord> DAGLayerChip<F> {
  pub fn construct(dag_config: DAGLayerConfig) -> Self {
    Self {
      dag_config,
      _marker: PhantomData,
    }
  }

  // A less efficient version of assigning tensors
  // Note: For our commitments, you should create fixed 'tensor columns'
  // and use the commitments to those (with a constant blinding factor)
  // as the commitment. Also note how you can do the fast commitments.
  pub fn assign_tensor_map(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensor: Array<F, IxDyn>,
  ) -> Result<AssignedTensor<F>, Error> {
    let tensors = layouter.assign_region(
      || "asssignment",
      |mut region| {
        let mut cell_idx = 0;

        let mut flat = vec![];
        for val in tensor.iter() {
          let row_idx = cell_idx / columns.len();
          let col_idx = cell_idx % columns.len();
          let cell = region
            .assign_advice(
              || "assignment",
              columns[col_idx],
              row_idx,
              || Value::known(*val),
            )
            .unwrap();
          flat.push(Rc::new(cell));
          cell_idx += 1;
        }
        let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();

        Ok(tensor)
      },
    )?;

    Ok(tensors)
  }

  pub fn assign_tensors_map(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
  ) -> Result<BTreeMap<i64, AssignedTensor<F>>, Error> {
    let tensors = layouter.assign_region(
      || "asssignment",
      |mut region| {
        let mut cell_idx = 0;
        let mut assigned_tensors = BTreeMap::new();

        for (tensor_idx, tensor) in tensors.iter() {
          let mut flat = vec![];
          for val in tensor.iter() {
            let row_idx = cell_idx / columns.len();
            let col_idx = cell_idx % columns.len();
            let cell = region
              .assign_advice(
                || "assignment",
                columns[col_idx],
                row_idx,
                || Value::known(*val),
              )
              .unwrap();
            flat.push(Rc::new(cell));
            cell_idx += 1;
          }
          let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();
          assigned_tensors.insert(*tensor_idx, tensor);
        }

        Ok(assigned_tensors)
      },
    )?;

    Ok(tensors)
  }

  pub fn tensor_map_to_vec(
    &self,
    tensor_map: &BTreeMap<i64, Array<CellRc<F>, IxDyn>>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
    let max_tensor_key = tensor_map
      .iter()
      .max_by_key(|(key, _)| *key)
      .unwrap()
      .0
      .clone();
    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap_or(smallest_tensor);
      tensors.push(tensor.clone());
    }

    Ok(tensors)
  }

  pub fn assign_tensors_vec(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let tensor_map = self
      .assign_tensors_map(
        layouter.namespace(|| "assign_tensors_map"),
        columns,
        tensors,
      )
      .unwrap();
    self.tensor_map_to_vec(&tensor_map)
  }

  // Initialize a tensor map with unassigned tensors
  pub fn initialized_unassigned_tensors_map(
    &self,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
  ) -> HashMap<usize, TensorAssignedOrUnassigned<F>> {
    tensors.iter().map(|(&ix, x)| (ix as usize, TensorAssignedOrUnassigned::Unassigned(x.clone()))).collect::<HashMap<usize, _>>()
  }

  // IMPORTANT: Assumes input tensors are in order. Output tensors can be in any order.
  pub fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<(HashMap<usize, AssignedTensor<F>>, Vec<AssignedTensor<F>>), Error> {
    // We only assign things in the tensor when we do not use them as weights.

    // let tensors = self.assign_tensors_vec(
    //   layouter.namespace(|| "assignment"),
    //   &gadget_config.columns,
    //   &tensors,
    // ).unwrap();

    let tensor_map = self.initialized_unassigned_tensors_map(tensors);

    // Compute the dag
    for (layer_idx, layer_config) in self.dag_config.ops.iter().enumerate() {
      let layer_type = &layer_config.layer_type;
      let inp_idxes = &self.dag_config.inp_idxes[layer_idx];
      let out_idxes = &self.dag_config.out_idxes[layer_idx];

      println!(
        "Processing layer {}, type: {:?}, inp_idxes: {:?}, out_idxes: {:?}, layer_params: {:?}",
        layer_idx, layer_type, inp_idxes, out_idxes, layer_config.layer_params
      );
      let raw_vec_inps = inp_idxes
        .iter()
        .map(|idx| tensor_map.get(idx).unwrap().clone())
        .collect::<Vec<_>>();
      
      // We try to assign all the vec inputs. Convert some things into fixed
      // tensors before we pass it through the compiler
      let mut vec_inps = vec![];
      // ZKML TODO: This is a decently pretty bad software architecture choice?
      // Probably is a cleaner way to do this.
      let mut flex_vec_inps = vec![];
      for (ix, &raw_inp) in raw_vec_inps.iter().enumerate() {
        // If we are in a "weights" section, then we will 
        if layer_type == &LayerType::Conv2D && ix == 2 {

        } else {

        }
        match raw_inp {
          &TensorAssignedOrUnassigned::Unassigned(inp) => {
            // Convert the vector to an assigned vector

          },
          &TensorAssignedOrUnassigned::Assigned(inp) => {
            vec_inps.push(inp);
          }
        }
        
      }
      
      let out = match layer_type {
        LayerType::Add => {
          let add_chip = AddChip {};
          add_chip.forward(
            layouter.namespace(|| "dag add"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::AvgPool2D => {
          let avg_pool_2d_chip = AvgPool2DChip {};
          avg_pool_2d_chip.forward(
            layouter.namespace(|| "dag avg pool 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::MaxPool2D => {
          let max_pool_2d_chip = MaxPool2DChip {
            marker: PhantomData::<F>,
          };
          max_pool_2d_chip.forward(
            layouter.namespace(|| "dag max pool 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::BatchMatMul => {
          let batch_mat_mul_chip = BatchMatMulChip {};
          batch_mat_mul_chip.forward(
            layouter.namespace(|| "dag batch mat mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Broadcast => {
          let broadcast_chip = BroadcastChip {};
          broadcast_chip.forward(
            layouter.namespace(|| "dag batch mat mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Conv2D => {
          let conv_2d_chip = Conv2DChip {
            config: layer_config.clone(),
            _marker: PhantomData,
          };
          conv_2d_chip.forward(
            layouter.namespace(|| "dag conv 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::DivFixed => {
          let div_fixed_chip = DivFixedChip {};
          div_fixed_chip.forward(
            layouter.namespace(|| "dag div"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::DivVar => {
          let div_var_chip = DivVarChip {};
          div_var_chip.forward(
            layouter.namespace(|| "dag div"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::FullyConnected => {
          let fc_chip = FullyConnectedChip {
            _marker: PhantomData,
            config: FullyConnectedConfig::construct(true),
          };
          fc_chip.forward(
            layouter.namespace(|| "dag fully connected"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Softmax => {
          let softmax_chip = SoftmaxChip {};
          softmax_chip.forward(
            layouter.namespace(|| "dag softmax"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Mean => {
          let mean_chip = MeanChip {};
          mean_chip.forward(
            layouter.namespace(|| "dag mean"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Pad => {
          let pad_chip = PadChip {};
          pad_chip.forward(
            layouter.namespace(|| "dag pad"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Permute => {
          let pad_chip = PermuteChip {};
          pad_chip.forward(
            layouter.namespace(|| "dag permute"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::SquaredDifference => {
          let squared_diff_chip = SquaredDiffChip {};
          squared_diff_chip.forward(
            layouter.namespace(|| "dag squared diff"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Rsqrt => {
          let rsqrt_chip = RsqrtChip {};
          rsqrt_chip.forward(
            layouter.namespace(|| "dag rsqrt"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Sqrt => {
          let sqrt_chip = SqrtChip {};
          sqrt_chip.forward(
            layouter.namespace(|| "dag sqrt"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Logistic => {
          let logistic_chip = LogisticChip {};
          logistic_chip.forward(
            layouter.namespace(|| "dag logistic"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Pow => {
          let pow_chip = PowChip {};
          pow_chip.forward(
            layouter.namespace(|| "dag logistic"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Tanh => {
          let tanh_chip = TanhChip {};
          tanh_chip.forward(
            layouter.namespace(|| "dag tanh"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Mul => {
          let mul_chip = MulChip {};
          mul_chip.forward(
            layouter.namespace(|| "dag mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Sub => {
          let sub_chip = SubChip {};
          sub_chip.forward(
            layouter.namespace(|| "dag sub"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Noop => {
          let noop_chip = NoopChip {};
          noop_chip.forward(
            layouter.namespace(|| "dag noop"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Transpose => {
          let transpose_chip = TransposeChip {};
          transpose_chip.forward(
            layouter.namespace(|| "dag transpose"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Reshape => {
          let reshape_chip = ReshapeChip {};
          reshape_chip.forward(
            layouter.namespace(|| "dag reshape"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::ResizeNN => {
          let resize_nn_chip = ResizeNNChip {};
          resize_nn_chip.forward(
            layouter.namespace(|| "dag resize nn"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Rotate => {
          let rotate_chip = RotateChip {};
          rotate_chip.forward(
            layouter.namespace(|| "dag rotate"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Concatenation => {
          let concat_chip = ConcatenationChip {};
          concat_chip.forward(
            layouter.namespace(|| "dag concatenation"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Pack => {
          let pack_chip = PackChip {};
          pack_chip.forward(
            layouter.namespace(|| "dag pack"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Split => {
          let split_chip = SplitChip {};
          split_chip.forward(
            layouter.namespace(|| "dag split"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Update => {
          let split_chip = UpdateChip {};
          split_chip.forward(
            layouter.namespace(|| "dag update"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Slice => {
          let slice_chip = SliceChip {};
          slice_chip.forward(
            layouter.namespace(|| "dag slice"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::MaskNegInf => {
          let mask_neg_inf_chip = MaskNegInfChip {};
          mask_neg_inf_chip.forward(
            layouter.namespace(|| "dag mask neg inf"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Square => {
          let square_chip = SquareChip {};
          square_chip.forward(
            layouter.namespace(|| "dag square"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
      };

      for (idx, tensor_idx) in out_idxes.iter().enumerate() {
        println!("Out {} shape: {:?}", idx, out[idx].shape());
        tensor_map.insert(*tensor_idx, out[idx].clone());
      }
    }

    let mut final_out = vec![];
    for idx in self.dag_config.final_out_idxes.iter() {
      final_out.push(tensor_map.get(idx).unwrap().clone());
    }

    let print_arr = if final_out.len() > 0 {
      &final_out[0]
    } else {
      if self.dag_config.ops.len() > 0 {
        let last_layer_idx = self.dag_config.ops.len() - 1;
        let out_idx = self.dag_config.out_idxes[last_layer_idx][0];
        tensor_map.get(&out_idx).unwrap()
      } else {
        tensor_map.get(&0).unwrap()
      }
    };

    let tmp = print_arr.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    print_assigned_arr("final out", &tmp.to_vec(), gadget_config.scale_factor);
    println!("final out idxes: {:?}", self.dag_config.final_out_idxes);

    let mut x = vec![];
    for cell in print_arr.iter() {
      cell.value().map(|v| {
        let bias = 1 << 60 as i64;
        let v_pos = *v + F::from(bias as u64);
        let v = convert_to_u64(&v_pos) as i64 - bias;
        x.push(v);
      });
    }
    if x.len() > 0 {
      let out_fname = "out.msgpack";
      let f = File::create(out_fname).unwrap();
      let mut buf = BufWriter::new(f);
      rmp_serde::encode::write_named(&mut buf, &x).unwrap();
    }

    Ok((tensor_map, final_out))
  }
}

impl<F: PrimeField + Ord> GadgetConsumer for DAGLayerChip<F> {
  // Special case: DAG doesn't do anything
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
