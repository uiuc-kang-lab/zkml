#![feature(int_roundings)]
use std::fs::File;
use std::io::{Error, Write};
use std::marker::PhantomData;

use halo2_proofs::{
    halo2curves::bn256::Fr,
    plonk::{ConstraintSystem, Circuit},
};
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

pub struct MyWrapper(ConstraintSystem<Fr>);

impl MyWrapper {
  pub fn new(inner: ConstraintSystem<Fr>) -> Self {
    MyWrapper(inner)
  }
  
  // Expose a public method that calls the private method
  pub fn permutation_required_degree(&self) {
      self.0.permutation().get_columns().len();
  }
}

fn load_constraint_from_circuit(_circuit: &ModelCircuit<Fr>) -> ConstraintSystem<Fr> {
    // create constraint system to collect custom gates
    let mut cs: ConstraintSystem<Fr> = Default::default();
    let _ = ModelCircuit::configure(&mut cs);
    cs
}

fn estimate_msm_time(k: i64, kzg_or_ipa: String) -> f64 {
  if kzg_or_ipa == "kzg" {
    match k {
      17 => 186596404.13333336,
      18 => 332822999.85,
      19 => 605177533.1,
      20 => 1121193479.2,
      21 => 2058248654.2,
      22 => 4051190316.6,
      23 => 7681884225.0,
      24 => 14132027162.7,
      _ => {
        println!("Warning: n_row is out of range for MSM time estimation. ");
        0.
      }
    }
  } else {
    match k {
      17 => 159672259.56420633,
      18 => 281278377.2,
      19 => 532171666.6,
      20 => 992834712.7,
      21 => 1893809104.4,
      22 => 3530607904.0,
      23 => 6589884079.3,
      24 => 12725500162.3,
      _ => {
        println!("Warning: n_row is out of range for MSM time estimation. ");
        0.
      }
    }
  }
}

fn estimate_fft_time(k: i64, kzg_or_ipa: String) -> f64 {
  if kzg_or_ipa == "kzg" {
    match k {
      17 => 10851875.196305115,
      18 => 22216662.588992067,
      19 => 46803234.09426586,
      20 => 104462216.81559524,
      21 => 239831734.73333335,
      22 => 549569116.7,
      23 => 1153836558.6,
      24 => 2448537633.3,
      _ => {
        println!("Warning: n_row is out of range for MSM time estimation. ");
        0.
      }
    }
  } else {
    match k {
      17 => 9997985.851182539,
      18 => 20777317.249984127,
      19 => 43106550.292050265,
      20 => 97406713.81198412,
      21 => 230773451.3333333,
      22 => 537737079.2,
      23 => 1132904375.0,
      24 => 2358579496.0,
      _ => {
        println!("Warning: n_row is out of range for MSM time estimation. ");
        0.
      }
    }
  }
}

fn cost_estimator(n_row: u64, n_instance: u64, n_advice: u64, n_lookup: u64, n_permutation: u64, max_degree:u64, kzg_or_ipa: String) -> f64 {
  let mut time: f64 = 0.0;
  let k = (n_row as f32).log2().ceil() as i16;
  let extended_k = k + ((max_degree-1) as f32).log2().ceil() as i16;

  let chunk = max_degree-2;
  let permutation_chunks = (n_permutation+chunk-1)/chunk;

  let n_fft = n_instance + n_advice + n_lookup * 3 + permutation_chunks;
  let n_coset_fft = n_instance + n_advice + n_lookup * 3 + permutation_chunks + 1;
  let n_msm: u64;
  if kzg_or_ipa == "kzg" {
    n_msm = n_instance + n_advice + n_lookup * 3 + permutation_chunks + max_degree - 1;
  } else {
    n_msm = n_instance + n_advice + n_lookup * 3 + permutation_chunks + max_degree;//+ 5;
  }
  println!("Num FFTs (n={}): {}", 1<<k, n_fft);
  println!("Num cosetFFTs (n={}): {}", 1<<extended_k, n_coset_fft);
  println!("Num MSMs (n={}): {}", 1<<k, n_msm);

  time += n_fft as f64 * estimate_fft_time(k as i64, kzg_or_ipa.clone());
  time += n_coset_fft as f64 * estimate_fft_time(extended_k as i64, kzg_or_ipa.clone());
  time += n_msm as f64 * estimate_msm_time(k as i64, kzg_or_ipa.clone());
  time
}

fn main() -> Result<(), Error> {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let kzg_or_ipa = std::env::args().nth(3).expect("kzg or ipa");

  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }

  let config = load_config_msgpack(&config_fname);
  let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);

  let num_cols = GADGET_CONFIG.lock().unwrap().num_cols as i64;
  println!("Num cols: {}", num_cols);

  let mut num_rows = circuit.num_random;

  // Number of rows from assignment
  let mut total_tensor_size = 0;
  for (_, tensor) in circuit.tensors.iter() {
    let num_elem = tensor.shape().iter().product::<usize>();
    total_tensor_size += num_elem;
  }

  num_rows += (total_tensor_size as i64).div_ceil(num_cols);
  println!("Total tensor + random size: {}", total_tensor_size);
  println!("Total rows for assignment: {}", num_rows);

  // Number of rows from layers
  for layer_config in circuit.dag_config.ops.iter() {
    println!("{:?}", layer_config);
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

    num_rows += layer_rows;
  }
  
  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);
  
  let cs: ConstraintSystem<Fr> = load_constraint_from_circuit(&circuit);
  let max_degree = cs.degree() as u64;
  let num_instance: u64 = cs.num_instance_columns() as u64;
  let num_advice: u64 = cs.num_advice_columns() as u64;
  let num_fixed: u64 = cs.num_fixed_columns() as u64;
  let num_lookup: u64 = cs.lookups().len() as u64;
  let num_permutation: u64 = cs.permutation().get_columns().len() as u64;
  println!("Max degree: {}", max_degree);
  println!("Num instance: {}", num_instance);
  println!("Num advice: {}", num_advice);
  println!("Num fixed: {}", num_fixed);
  println!("Num lookup: {}", num_lookup);
  println!("Num permutation: {}", num_permutation);


  let time_cost = cost_estimator(num_rows as u64, num_instance, num_advice, num_lookup, num_permutation, max_degree, kzg_or_ipa);
  let k = (num_rows as f32).log2().ceil() as i64;
  
  let path = "k.tmp";
  let mut output = File::create(path)?;
  write!(output, "{}", k)?;
  
  println!("Total number of rows: {}", num_rows);
  println!("Total time cost (esitmated): {} (ns)", time_cost);
  Ok(())
}
