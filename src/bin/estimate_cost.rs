#![feature(int_roundings)]
pub use rayon::{current_num_threads, scope, Scope};
use serde_json::Value;
use rmp_serde::decode::from_read;
use std::fs::File;
use std::io::Error;
use std::marker::PhantomData;
use std::panic;
use halo2_proofs::{
    halo2curves::bn256::{G1, Fr},
    plonk::{ConstraintSystem, Circuit},
    dev::cost::CircuitCost,
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
};

// for debugging
fn circuit_cost_without_permutation(circuit: ModelCircuit<Fr>, k: u64) -> u64 {
    let mut k = k;
    loop {
      let result = panic::catch_unwind(|| {
        CircuitCost::<G1, ModelCircuit<Fr>>::measure((k as u32).try_into().unwrap(), &circuit)
      });
      match result {
        Ok(cost) => {
          break;
        }
        Err(_) => {
          println!("k = {} is not enough", k);
          k += 1;
        }
      }
    }
    k as u64
  }

fn load_constraint_from_circuit(_circuit: &ModelCircuit<Fr>) -> ConstraintSystem<Fr> {
    // create constraint system to collect custom gates
    let mut cs: ConstraintSystem<Fr> = Default::default();
    let _ = ModelCircuit::configure(&mut cs);
    cs
}

fn sum_gates_degree<F: halo2_proofs::arithmetic::Field>(cs: ConstraintSystem<F>) -> u64 {
  // The permutation argument will serve alongside the gates, so must be accounted for.
  let mut degree: u64 = 3; 

  // The lookup argument also serves alongside the gates and must be accounted for.
  let lookups = cs.lookups();
  let input_degree = 1;
  let table_degree = 1;
  for lookup in lookups {
    for expr in lookup.input_expressions().iter() {
      degree += std::cmp::max(input_degree, expr.degree()) as u64;
      //println!("Input degree: {}", expr.degree())
    }
    for expr in lookup.table_expressions().iter() {
      degree += std::cmp::max(table_degree, expr.degree()) as u64;
      //println!("Table degree: {}", expr.degree())
    }
  }

  // Account for each gate to ensure our quotient polynomial is the correct degree and that our extended domain is the right size.
  for gate in cs.gates() {
    for poly in gate.polynomials().iter() {
      degree += poly.degree() as u64;
      //println!("Poly degree: {}", poly.degree())
    }
  }
  
  degree
}

fn estimate_time(k: i64, kzg_or_ipa: String, oper_type: String, bench_statistics: &Value) -> f64 {
  let key = format!("{}_{}", kzg_or_ipa, oper_type);
  if oper_type == "mul" {
    bench_statistics[key].as_f64().unwrap() * (1<<k) as f64
  } 
  else {
    if let Some(value) = bench_statistics[key].get(k.to_string()) {
      value.as_f64().unwrap()
    } else {
      println!("Warning: k is out of range for {} time estimation. ", oper_type);
      0.
    }
  }
  
}

fn cost_estimator<F: halo2_proofs::arithmetic::Field>(k: u64, cs: ConstraintSystem<F>, kzg_or_ipa: String, bench_statistics: &Value) -> f64 {
  let max_degree = cs.degree() as u64;
  let n_instance: u64 = cs.num_instance_columns() as u64;
  let n_advice: u64 = cs.num_advice_columns() as u64;
  let n_lookup: u64 = cs.lookups().len() as u64;
  let n_permutation: u64 = cs.permutation().get_columns().len() as u64;
  let mut sum_degree_num = sum_gates_degree(cs);
  
  //let num_threads = current_num_threads() as f64;
  let num_threads = 32 as f64;
  println!("Num threads: {}", num_threads);
  let mut time: f64 = 0.0;
  let k = k as i16;
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
  // println!("Num FFTs (n={}): {}", 1<<k, n_fft);
  // println!("Num cosetFFTs (n={}): {}", 1<<extended_k, n_coset_fft);
  // println!("Num MSMs (n={}): {}", 1<<k, n_msm);

  // Main costs (FFTs and MSMs)
  time += n_fft as f64 * estimate_time(k as i64, kzg_or_ipa.clone(), "fft".to_string(), bench_statistics);
  time += n_coset_fft as f64 * estimate_time(extended_k as i64, kzg_or_ipa.clone(), "fft".to_string(), bench_statistics);
  time += n_msm as f64 * estimate_time(k as i64, kzg_or_ipa.clone(), "msm".to_string(), bench_statistics);
  println!("Total time cost (FFT + MSM): {} (ns)", time);
  // Additional costs (permutations and multiplications)
  time += n_lookup as f64 * estimate_time(k as i64, kzg_or_ipa.clone(), "permute".to_string(), bench_statistics);
  println!("Total time cost (FFT + MSM + Permute): {} (ns)", time);
  sum_degree_num += permutation_chunks * 14 + n_lookup * 19;
  time += sum_degree_num as f64 * estimate_time(k as i64, kzg_or_ipa.clone(), "mul".to_string(), bench_statistics) / num_threads;
  println!("Total time cost (FFT + MSM + Permute + Mul): {} (ns)", time);

  time
}

fn main() -> Result<(), Error> {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let kzg_or_ipa = std::env::args().nth(3).expect("kzg or ipa");
  
  // Open the Msgpack file
  let mut bench_file = File::open("summary.msgpack").expect("Failed to open file");

  // Deserialize the data into a serde_json::Value
  let bench_statistics: Value = from_read(&mut bench_file).expect("Failed to deserialize data");

  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }

  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);

  let num_cols = GADGET_CONFIG.lock().unwrap().num_cols as i64;
  println!("Num cols: {}", num_cols);

  let num_constants = circuit.num_random + 5; // num_randoms + vec![0 as i64, 1, sf as i64, min_val, max_val];
  let mut num_rows = num_constants.div_ceil(num_cols);

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
  
  let cs: ConstraintSystem<Fr> = load_constraint_from_circuit(&circuit);
  
  // blinding factor
  let blinding_factor = cs.minimum_rows() as i64;
  println!("Blinding factor: {}", blinding_factor);
  num_rows += blinding_factor;
  
  let k = (num_rows as f32).log2().ceil() as u64;
  //let k = circuit_cost_without_permutation(circuit.clone(), k);
  
  // Print out some stats
  let max_degree = cs.degree() as u64;
  let num_instance: u64 = cs.num_instance_columns() as u64;
  let num_advice: u64 = cs.num_advice_columns() as u64;
  let num_fixed: u64 = cs.num_fixed_columns() as u64;
  let num_lookup: u64 = cs.lookups().len() as u64;
  let num_permutation: u64 = cs.permutation().get_columns().len() as u64;
  let num_gates = cs.gates().len() as u64;
  let sum_degrees = sum_gates_degree(cs.clone());
  
  println!("Max degree: {}", max_degree);
  println!("Num instance: {}", num_instance);
  println!("Num advice: {}", num_advice);
  println!("Num fixed: {}", num_fixed);
  println!("Num lookup: {}", num_lookup);
  println!("Num permutation: {}", num_permutation);
  println!("Num gates: {}", num_gates);
  println!("Sum degrees: {}", sum_degrees);
  // The above println scripts are for debugging purposes only
  
  let time_cost = cost_estimator(k, cs, kzg_or_ipa, &bench_statistics);
  
  println!("Optimal k: {}", k);
  println!("Total number of rows: {}", num_rows);
  println!("Total time cost (esitmated): {} (ns)", time_cost);
  Ok(())
}
