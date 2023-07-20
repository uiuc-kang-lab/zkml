#![feature(int_roundings)]

use std::{marker::PhantomData, collections::HashMap};

use halo2_gadgets::ecc::chip::H;
use halo2_proofs::halo2curves::bn256::Fr;
use zkml::{
  layers::{
    arithmetic::{add::AddChip, div_var::DivVarChip, mul::MulChip, sub::SubChip},
    avg_pool_2d::AvgPool2DChip,
    batch_mat_mul::BatchMatMulChip,
    conv2d::Conv2DChip,
    div_fixed::DivFixedChip,
    fc::fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    layer::{Layer, LayerType},
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
  model::{ModelCircuit, GADGET_CONFIG},
  utils::loader::load_config_msgpack,
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");

  let config = load_config_msgpack(&config_fname);
  let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);
  let num_cols = GADGET_CONFIG.lock().unwrap().num_cols as i64;

  let mut num_rows = circuit.num_random;

  // Number of rows from assignment
  // println!("tensors {:?}", circuit.tensors);
  let mut total_tensor_size = 0;
  // for (_, tensor) in circuit.tensors.iter() {
  //   println!("in tensor {:?}", tensor);
  //   let num_elem = tensor.shape().iter().product::<usize>();
  //   total_tensor_size += num_elem;
  // }
  // println!("TOTAL TENSOR {:?}")

  num_rows += (total_tensor_size as i64).div_ceil(num_cols);
  println!("Total tensor + random size: {}", total_tensor_size);
  println!("Total rows for assignment: {}", num_rows);

  let mut hash_map = HashMap::new();

  println!("NUM ROWS BEFORE {:?}", num_rows);

  // Number of rows from layers
  for layer_config in circuit.dag_config.ops.iter() {
    let layer_rows = match layer_config.layer_type {
      LayerType::Add => {
        let chip = AddChip {};
        <AddChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::AvgPool2D => {
        let chip = AvgPool2DChip {};
        <AvgPool2DChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::BatchMatMul => {
        let chip = BatchMatMulChip {};
        <BatchMatMulChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Broadcast => {
        let chip = BroadcastChip {};
        <BroadcastChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Concatenation => {
        let chip = ConcatenationChip {};
        <ConcatenationChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Conv2D => {
        let chip = Conv2DChip {
          config: layer_config.clone(),
          _marker: PhantomData,
        };
        <Conv2DChip<Fr> as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::DivFixed => {
        let chip = DivFixedChip {};
        <DivFixedChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::DivVar => {
        let chip = DivVarChip {};
        <DivVarChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::FullyConnected => {
        let fc_chip = FullyConnectedChip {
          _marker: PhantomData,
          config: FullyConnectedConfig::construct(true),
        };
        <FullyConnectedChip<Fr> as Layer<Fr>>::num_rows(&fc_chip, &layer_config, num_cols)
      }
      LayerType::Logistic => {
        let chip = LogisticChip {};
        <LogisticChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::MaskNegInf => {
        let chip = MaskNegInfChip {};
        <MaskNegInfChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::MaxPool2D => {
        let chip = MaxPool2DChip {
          marker: PhantomData,
        };
        <MaxPool2DChip<Fr> as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Mean => {
        let chip = MeanChip {};
        <MeanChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Mul => {
        let chip = MulChip {};
        <MulChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Noop => {
        let chip = NoopChip {};
        <NoopChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Pack => {
        let chip = PackChip {};
        <PackChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Pad => {
        let chip = PadChip {};
        <PadChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Permute => {
        let chip = PermuteChip {};
        <PermuteChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Pow => {
        let chip = PowChip {};
        <PowChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Reshape => {
        let chip = ReshapeChip {};
        <ReshapeChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::ResizeNN => {
        let chip = ResizeNNChip {};
        <ResizeNNChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Rotate => {
        let chip = RotateChip {};
        <RotateChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Rsqrt => {
        let chip = RsqrtChip {};
        <RsqrtChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Slice => {
        let chip = SliceChip {};
        <SliceChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Softmax => {
        let chip = SoftmaxChip {};
        <SoftmaxChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Split => {
        let chip = SplitChip {};
        <SplitChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Sqrt => {
        let chip = SqrtChip {};
        <SqrtChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Square => {
        let chip = SquareChip {};
        <SquareChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::SquaredDifference => {
        let chip = SquaredDiffChip {};
        <SquaredDiffChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Sub => {
        let chip = SubChip {};
        <SubChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Tanh => {
        let chip = TanhChip {};
        <TanhChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Transpose => {
        let chip = TransposeChip {};
        <TransposeChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
      LayerType::Update => {
        let chip = UpdateChip {};
        <UpdateChip as Layer<Fr>>::num_rows(&chip, &layer_config, num_cols)
      }
    };
    let mut row = hash_map.entry(layer_config.layer_type).or_insert(0);
    *row += layer_rows;

    num_rows += layer_rows;
  }

  println!("Total number of rows: {:?}", hash_map);
  println!("Total number of rows: {}", num_rows);
  println!(
    "Total number of rows after blinding (estimated): {}",
    num_rows + 15
  );
}
