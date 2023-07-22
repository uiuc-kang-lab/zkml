use std::{collections::{HashMap, BTreeMap}, fs::File, io::BufWriter, marker::PhantomData, rc::Rc};

use halo2_proofs::{circuit::{Layouter, Value, AssignedCell}, halo2curves::ff::PrimeField, plonk::{Error, MatrixConfig, Advice, Column, Assigned}};
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
#[derive(Debug)]
pub struct MatmulAssignment<F: PrimeField> {
  tensor: Array<F, IxDyn>,
  input_vec: Vec<AssignedCell<F, F>>,
  output_vec: Vec<AssignedCell<F, F>>,
}


/// The final matrix
pub struct MatrixOutput<F: PrimeField> {
  config: MatrixConfig,
  filled_index: usize,
  /// stacked input vector
  input_vector: Vec<AssignedCell<F, F>>,
  /// output vector to be accumulated
  output_vectors: Vec<Vec<AssignedCell<F, F>>>,
  /// stacked tensor
  tensor: Array<F, IxDyn>,
}

impl<F: PrimeField> MatrixOutput<F> {
  /// Create a new cqlin matrix output row
  pub fn new(config: MatrixConfig) -> Self {
    let h = 1 << config.l;
    let w = 1 << config.k;
    let input_vector = vec![];
    let output_vectors = vec![];
    let tensor = Array::from_elem(IxDyn(&[h, w]), F::ZERO);

    MatrixOutput {
      config,
      filled_index: 0,
      input_vector,
      output_vectors,
      tensor,
    }
  }

  /// Push a new tensor to this row. If the row is true, return false (should create a new row if this is the case)
  pub fn check_availability(&self, tensor: &Array<F, IxDyn>) -> bool {
    let ttensor = tensor.t().to_owned();
    let tensor_h = ttensor.shape()[0];

    if tensor_h + self.filled_index > self.tensor.shape()[0] {
      false
    } else {
    true
    }
  }

  pub fn push_tensor(&mut self, tensor: Array<F, IxDyn>, input_vec: Vec<AssignedCell<F, F>>, output_vec: Vec<AssignedCell<F, F>>) -> bool {
    let ttensor = tensor.t().to_owned();
    let tensor_h = ttensor.shape()[0];
    let tensor_w = ttensor.shape()[1];

    for i in 0..tensor_h {
      for j in 0..tensor_w {
        self.tensor[[self.filled_index + i, j]] = ttensor[[i, j]];
      }
    }
    self.filled_index += tensor_h;

    self.input_vector.extend(input_vec.into_iter());
    self.output_vectors.push(output_vec);

    true
  }
}

/// The vector 
/// We also want to actually 
pub struct VectorEngine<F: PrimeField> {
  pub assignments: HashMap<MatrixConfig, Vec<MatmulAssignment<F>>>,
}

impl<F: PrimeField> VectorEngine<F> {
  // So right now, for our batchmul thing, we are essentially doing a gate where we compute the frievalded output. Right now, we do a batch size
  // 1. Create the frievalds' ready
  // 
  pub fn assign_matmul(
    &mut self,
    matrix_log: &MatrixLog,
    weight: &Array<F, IxDyn>,
    input_vec: &Vec<&AssignedCell<F, F>>,
    output_vec: &Vec<&AssignedCell<F, F>>,
  ) {
    assert_eq!(input_vec.len(), weight.shape()[1]);
    assert_eq!(output_vec.len(), weight.shape()[0]);

    let smallest_config = matrix_log.select_log(weight.shape()[0]);
    let mat_log_vec = self.assignments.entry(smallest_config).or_insert(vec![]);
    mat_log_vec.push(MatmulAssignment {
      tensor: weight.clone(),
      input_vec: input_vec.iter().cloned().cloned().collect::<Vec<_>>(),
      output_vec: output_vec.iter().cloned().cloned().collect::<Vec<_>>()
    });
    println!("MATMUL PROGRESS {:?}", mat_log_vec);
  }

  pub fn generate_column_assignments(self) -> HashMap<MatrixConfig, Vec<MatrixOutput<F>>> {
    let mut matrix_outputs = HashMap::new();
    for (config, assignments) in self.assignments.into_iter() {
      let mut curr_matrix_output = MatrixOutput::new(config);
      let mut config_matrix_outputs = vec![];

      for assignment in assignments.into_iter() {
        let MatmulAssignment {
          tensor,
          input_vec,
          output_vec
        } = assignment;
        // If cqlin column unavailable, create a new column.
        if !curr_matrix_output.check_availability(&tensor) {
          config_matrix_outputs.push(curr_matrix_output);
          curr_matrix_output = MatrixOutput::new(config);
        }
        curr_matrix_output.push_tensor(tensor, input_vec, output_vec);
      }
      config_matrix_outputs.push(curr_matrix_output);
      matrix_outputs.insert(config, config_matrix_outputs);
    }

    matrix_outputs
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

#[derive(Debug, Clone)]
pub enum TensorAssignedOrUnassigned<F: PrimeField> {
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
  pub fn assign_tensor(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensor: &Array<F, IxDyn>,
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
    vector_engine: &mut VectorEngine<F>,
   ) -> Result<(HashMap<usize, TensorAssignedOrUnassigned<F>>, Vec<TensorAssignedOrUnassigned<F>>), Error> {
    // We only assign things in the tensor when we do not use them as weights.

    // let tensors = self.assign_tensors_vec(
    //   layouter.namespace(|| "assignment"),
    //   &gadget_config.columns,
    //   &tensors,
    // ).unwrap();

    let mut tensor_map = self.initialized_unassigned_tensors_map(tensors);

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
      for (ix, raw_inp) in raw_vec_inps.iter().enumerate() {
        // If we are in a "weights" section, then we will 
        // In this case, never assign
        if layer_type == &LayerType::Conv2D && ix == 1 {
          flex_vec_inps.push(raw_inp.clone());
        }
        // } else {

        match raw_inp {
          TensorAssignedOrUnassigned::Assigned(inp) => {
            vec_inps.push(inp.clone());
          },
          TensorAssignedOrUnassigned::Unassigned(inp) => {
            // If it is some random component, we replace and send it.
            let assigned_inp = self.assign_tensor(layouter.namespace(|| ""), &gadget_config.columns, inp)?;
            tensor_map.insert(inp_idxes[ix], TensorAssignedOrUnassigned::Assigned(assigned_inp.clone()));
            vec_inps.push(assigned_inp);
          }
        }

        // }
      }
      
      let out = match layer_type {
        LayerType::Add => {
          let add_chip = AddChip {};
          add_chip.forward(
            layouter.namespace(|| "dag add"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::AvgPool2D => {
          let avg_pool_2d_chip = AvgPool2DChip {};
          avg_pool_2d_chip.forward(
            layouter.namespace(|| "dag avg pool 2d"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::MaxPool2D => {
          let max_pool_2d_chip = MaxPool2DChip {
            marker: PhantomData::<F>,
          };
          max_pool_2d_chip.forward(
            layouter.namespace(|| "dag max pool 2d"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::BatchMatMul => {
          let batch_mat_mul_chip = BatchMatMulChip {};
          batch_mat_mul_chip.forward(
            layouter.namespace(|| "dag batch mat mul"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Broadcast => {
          let broadcast_chip = BroadcastChip {};
          broadcast_chip.forward(
            layouter.namespace(|| "dag batch mat mul"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
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
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::DivFixed => {
          let div_fixed_chip = DivFixedChip {};
          div_fixed_chip.forward(
            layouter.namespace(|| "dag div"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::DivVar => {
          let div_var_chip = DivVarChip {};
          div_var_chip.forward(
            layouter.namespace(|| "dag div"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
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
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Softmax => {
          let softmax_chip = SoftmaxChip {};
          softmax_chip.forward(
            layouter.namespace(|| "dag softmax"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Mean => {
          let mean_chip = MeanChip {};
          mean_chip.forward(
            layouter.namespace(|| "dag mean"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Pad => {
          let pad_chip = PadChip {};
          pad_chip.forward(
            layouter.namespace(|| "dag pad"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Permute => {
          let pad_chip = PermuteChip {};
          pad_chip.forward(
            layouter.namespace(|| "dag permute"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::SquaredDifference => {
          let squared_diff_chip = SquaredDiffChip {};
          squared_diff_chip.forward(
            layouter.namespace(|| "dag squared diff"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Rsqrt => {
          let rsqrt_chip = RsqrtChip {};
          rsqrt_chip.forward(
            layouter.namespace(|| "dag rsqrt"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Sqrt => {
          let sqrt_chip = SqrtChip {};
          sqrt_chip.forward(
            layouter.namespace(|| "dag sqrt"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Logistic => {
          let logistic_chip = LogisticChip {};
          logistic_chip.forward(
            layouter.namespace(|| "dag logistic"),
            &vec_inps,
            &flex_vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Pow => {
          let pow_chip = PowChip {};
          pow_chip.forward(
            layouter.namespace(|| "dag logistic"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Tanh => {
          let tanh_chip = TanhChip {};
          tanh_chip.forward(
            layouter.namespace(|| "dag tanh"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Mul => {
          let mul_chip = MulChip {};
          mul_chip.forward(
            layouter.namespace(|| "dag mul"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Sub => {
          let sub_chip = SubChip {};
          sub_chip.forward(
            layouter.namespace(|| "dag sub"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Noop => {
          let noop_chip = NoopChip {};
          noop_chip.forward(
            layouter.namespace(|| "dag noop"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Transpose => {
          let transpose_chip = TransposeChip {};
          transpose_chip.forward(
            layouter.namespace(|| "dag transpose"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Reshape => {
          let reshape_chip = ReshapeChip {};
          reshape_chip.forward(
            layouter.namespace(|| "dag reshape"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::ResizeNN => {
          let resize_nn_chip = ResizeNNChip {};
          resize_nn_chip.forward(
            layouter.namespace(|| "dag resize nn"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Rotate => {
          let rotate_chip = RotateChip {};
          rotate_chip.forward(
            layouter.namespace(|| "dag rotate"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Concatenation => {
          let concat_chip = ConcatenationChip {};
          concat_chip.forward(
            layouter.namespace(|| "dag concatenation"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Pack => {
          let pack_chip = PackChip {};
          pack_chip.forward(
            layouter.namespace(|| "dag pack"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Split => {
          let split_chip = SplitChip {};
          split_chip.forward(
            layouter.namespace(|| "dag split"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Update => {
          let split_chip = UpdateChip {};
          split_chip.forward(
            layouter.namespace(|| "dag update"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Slice => {
          let slice_chip = SliceChip {};
          slice_chip.forward(
            layouter.namespace(|| "dag slice"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::MaskNegInf => {
          let mask_neg_inf_chip = MaskNegInfChip {};
          mask_neg_inf_chip.forward(
            layouter.namespace(|| "dag mask neg inf"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
        LayerType::Square => {
          let square_chip = SquareChip {};
          square_chip.forward(
            layouter.namespace(|| "dag square"),
            &vec_inps,
            &flex_vec_inps,
             constants,
            gadget_config.clone(),
            &layer_config,
            vector_engine,
          )?
        }
      };

      for (idx, tensor_idx) in out_idxes.iter().enumerate() {
        println!("Out {} shape: {:?}", idx, out[idx].shape());
        tensor_map.insert(*tensor_idx, TensorAssignedOrUnassigned::Assigned(out[idx].clone()));
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

    // let tmp = print_arr.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    // print_assigned_arr("final out", &tmp.to_vec(), gadget_config.scale_factor);
    println!("final out idxes: {:?}", self.dag_config.final_out_idxes);

    // let mut x = vec![];
    // for cell in print_arr.iter() {
    //   cell.value().map(|v| {
    //     let bias = 1 << 60 as i64;
    //     let v_pos = *v + F::from(bias as u64);
    //     let v = convert_to_u64(&v_pos) as i64 - bias;
    //     x.push(v);
    //   });
    // }
    // if x.len() > 0 {
    //   let out_fname = "out.msgpack";
    //   let f = File::create(out_fname).unwrap();
    //   let mut buf = BufWriter::new(f);
    //   rmp_serde::encode::write_named(&mut buf, &x).unwrap();
    // }

    Ok((tensor_map, final_out))
    // Ok(())
  }
}

impl<F: PrimeField + Ord> GadgetConsumer for DAGLayerChip<F> {
  // Special case: DAG doesn't do anything
  fn used_gadgets(&self, _layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
