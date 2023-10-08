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

fn circuit_cost_without_permutation(circuit: ModelCircuit<Fr>, k: u64) -> u64 {
    let mut k = k;
    loop {
      let result = panic::catch_unwind(|| {
        CircuitCost::<G1, ModelCircuit<Fr>>::measure((k as u32).try_into().unwrap(), &circuit)
      });
      match result {
        Ok(_) => {
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

fn simulate_evaluate_h<F: halo2_proofs::arithmetic::Field>(k: u64, cs: ConstraintSystem<F>) -> (u64, u64) {
  let mut add_num: u64 = 0;
  let mut mul_num: u64 = 0;

  // custom gates
  let (cg_add_num, cg_mul_num) = cs
    .gates()
    .iter()
    .flat_map(|gate| {
        gate.polynomials().iter().map(|poly| {
            poly.evaluate(
                &|_| (0, 0),
                &|_| (0, 0),
                &|_| (0, 0),
                &|_| (0, 0),
                &|_| (0, 0),
                &|_| (0, 0),
                &|(a_a, a_m)| (a_a + 1, a_m),
                &|(a_a, a_m), (b_a, b_m)| (a_a + b_a + 1, a_m + b_m),
                &|(a_a, a_m), (b_a, b_m)| (a_a + b_a, a_m + b_m + 1),
                &|(a_a, a_m), _| (a_a, a_m + 1),
            )
        })
    })
    .fold((0, 0), |(acc_a, acc_m), (a, m)| {
        (acc_a + a, acc_m + m)
    });

  add_num += cg_add_num;
  mul_num += cg_mul_num;

  // Permutations
  let chunk_len = (cs.degree() - 2) as u64;
  let num_perm_slices = (cs.permutation().get_columns().len() as u64 + chunk_len - 1) / chunk_len;

  if num_perm_slices > 0 {
    // Enforce only for the first set.
    // value(X) = value(X) * y + l_0(X) * (1 - z_0(X))
    add_num += 2;
    mul_num += 2;
    // Enforce only for the last set.
    // value(X) = value(X) * y + l_last(X) * (z_l(X)^2 - z_l(X))
    add_num += 2;
    mul_num += 3;
    // Except for the first set, enforce.
    // value(X) = value(X) * y + l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X))
    add_num += 2 * (num_perm_slices - 1);
    mul_num += 2 * (num_perm_slices - 1);
    // delta_start * beta_start
    mul_num += 1;
    // And for all the sets we enforce:
    // (1 - (l_last(X) + l_blind(X))) * (
    //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
    // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
    // )
    // Calculate left = z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
    let mut tmp_add_num = 0;
    let mut tmp_mul_num = 0;
    tmp_add_num += 2 * chunk_len;
    tmp_mul_num += 2 * chunk_len;
    // Calculate right = z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma), current_delta *= DELTA
    tmp_add_num += 2 * chunk_len;
    tmp_mul_num += chunk_len;
    tmp_mul_num += chunk_len;
    // value(X) = value(X) * y + (1 - (l_last(X) + l_blind(X))) * (
    //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
    // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
    // ).
    tmp_add_num += 2;
    tmp_mul_num += 2;
    add_num += tmp_add_num * num_perm_slices;
    mul_num += tmp_mul_num * num_perm_slices;
    // beta_term *= &extended_omega;
    mul_num += 1;
  }

  // Lookups
  let (lk_inp_add_num, lk_inp_mul_num) = cs
    .lookups()
    .iter()
    .flat_map(|lookup| {
      lookup
        .input_expressions()
        .iter()
        .map(|expr| { expr.evaluate(
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|(a_a, a_m)| (a_a + 1, a_m),
          &|(a_a, a_m), (b_a, b_m)| (a_a + b_a + 1, a_m + b_m),
          &|(a_a, a_m), (b_a, b_m)| (a_a + b_a, a_m + b_m + 1),
          &|(a_a, a_m), _| (a_a, a_m + 1),
        )
      })
    })
    .fold((0, 0), |(acc_a, acc_m), (a, m)| {
      (acc_a + a, acc_m + m)
    });
  
    let (lk_tab_add_num, lk_tab_mul_num) = cs
    .lookups()
    .iter()
    .flat_map(|lookup| {
      lookup
        .table_expressions()
        .iter()
        .map(|expr| { expr.evaluate(
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|_| (0, 0),
          &|(a_a, a_m)| (a_a + 1, a_m),
          &|(a_a, a_m), (b_a, b_m)| (a_a + b_a + 1, a_m + b_m),
          &|(a_a, a_m), (b_a, b_m)| (a_a + b_a, a_m + b_m + 1),
          &|(a_a, a_m), _| (a_a, a_m + 1),
        )
      })
    })
    .fold((0, 0), |(acc_a, acc_m), (a, m)| {
      (acc_a + a, acc_m + m)
    });

  add_num += lk_inp_add_num + lk_tab_add_num;
  mul_num += lk_inp_mul_num + lk_tab_mul_num;

  let num_lookups = cs.lookups().len() as u64;
  // a_minus_s
  add_num += num_lookups;
  // value(X) = value(X) * y + l_0(X) * (1 - z(X))
  add_num += 2 * num_lookups;
  mul_num += 2 * num_lookups;
  // value(X) = value(X) * y + l_last(X) * (z(X)^2 - z(X))
  add_num += 2 * num_lookups;
  mul_num += 3 * num_lookups;
  // value(X) = value(X) * y + (1 - (l_last(X) + l_blind(X))) * (
  //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
  //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
  //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
  // )
  add_num += 4 * num_lookups;
  mul_num += 5 * num_lookups;
  // value(X) = value(X) * y + l_0(X) * (a'(X) - s'(X))
  add_num += 1 * num_lookups;
  mul_num += 2 * num_lookups;
  // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
  add_num += 2 * num_lookups;
  mul_num += 3 * num_lookups;
  
  // Align the constraints by different powers of y.
  let max_degree = cs.degree() as u64;
  let extended_k = k + ((max_degree-1) as f32).log2().ceil() as u64;
  let num_clusters = extended_k - k + 1;
  add_num += num_clusters;
  mul_num += num_clusters;

  (add_num, mul_num)
}

fn estimate_time(k: i64, kzg_or_ipa: String, oper_type: String, bench_statistics: &Value) -> f64 {
  let key = format!("{}_{}", kzg_or_ipa, oper_type);
  if let Some(value) = bench_statistics[key].get(k.to_string()) {
    value.as_f64().unwrap()
  } else {
    println!("Warning: k is out of range for {} time estimation. ", oper_type);
    0.
  }
}

fn cost_estimator<F: halo2_proofs::arithmetic::Field>(k: u64, cs: ConstraintSystem<F>, kzg_or_ipa: String, bench_statistics: &Value) -> f64 {
  let max_degree = cs.degree() as u64;
  let n_instance: u64 = cs.num_instance_columns() as u64;
  let n_advice: u64 = cs.num_advice_columns() as u64;
  let n_lookup: u64 = cs.lookups().len() as u64;
  let n_permutation: u64 = cs.permutation().get_columns().len() as u64;
  let (add_num, mul_num) = simulate_evaluate_h(k, cs);

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
  time += add_num as f64 * estimate_time(extended_k as i64, kzg_or_ipa.clone(), "add".to_string(), bench_statistics);
  time += mul_num as f64 * estimate_time(extended_k as i64, kzg_or_ipa.clone(), "mul".to_string(), bench_statistics);
  println!("Total time cost (FFT + MSM + Permute + Arithmetic): {} (ns)", time);

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
  let k = circuit_cost_without_permutation(circuit.clone(), k);
  
  // Print out some stats
  let max_degree = cs.degree() as u64;
  let num_instance: u64 = cs.num_instance_columns() as u64;
  let num_advice: u64 = cs.num_advice_columns() as u64;
  let num_fixed: u64 = cs.num_fixed_columns() as u64;
  let num_lookup: u64 = cs.lookups().len() as u64;
  let num_permutation: u64 = cs.permutation().get_columns().len() as u64;
  let num_gates = cs.gates().len() as u64;
  
  println!("Max degree: {}", max_degree);
  println!("Num instance: {}", num_instance);
  println!("Num advice: {}", num_advice);
  println!("Num fixed: {}", num_fixed);
  println!("Num lookup: {}", num_lookup);
  println!("Num permutation: {}", num_permutation);
  println!("Num gates: {}", num_gates);
  // The above println scripts are for debugging purposes only
  
  let time_cost = cost_estimator(k, cs, kzg_or_ipa, &bench_statistics);
  
  println!("Optimal k: {}", k);
  println!("Total number of rows: {}", num_rows);
  println!("Total time cost (esitmated): {} (ns)", time_cost);
  Ok(())
}
