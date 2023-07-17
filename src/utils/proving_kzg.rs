use std::{
  fs::File,
  io::{BufReader, Write},
  path::Path,
  time::Instant, collections::HashMap,
};

use halo2_proofs::{
  dev::MockProver,
  halo2curves::bn256::{Bn256, Fr, G1Affine},
  plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, VerifyingKey, MatrixConfig},
  poly::{
    commitment::Params,
    kzg::{
      commitment::{KZGCommitmentScheme, ParamsKZG},
      multiopen::{ProverSHPLONK, VerifierSHPLONK},
      strategy::SingleStrategy,
    },
  },
  transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
  },
  SerdeFormat,
};

use crate::{model::ModelCircuit, utils::helpers::get_public_values};

pub fn get_kzg_params(params_dir: &str, degree: u32, max_cqlin_degree: u32, max_cq_degree: u32) -> ParamsKZG<Bn256> {
  let rng = rand::thread_rng();
  let path = format!("{}/{}.params", params_dir, degree);
  let params_path = Path::new(&path);
  if File::open(&params_path).is_err() {
    let params = ParamsKZG::<Bn256>::setup(degree, max_cqlin_degree, max_cq_degree, rng);
    let mut buf = Vec::new();

    params.write(&mut buf).expect("Failed to write params");
    let mut file = File::create(&params_path).expect("Failed to create params file");
    file
      .write_all(&buf[..])
      .expect("Failed to write params to file");
  }

  let mut params_fs = File::open(&params_path).expect("couldn't load params");
  let params = ParamsKZG::<Bn256>::read(&mut params_fs).expect("Failed to read params");
  params
}

pub fn get_cqlin_kzg_params(params: &ParamsKZG<Bn256>, cqlin_params_dir: &str, lengths: Vec<usize>) -> HashMap<usize, ParamsKZG<Bn256>> {
  let rng = rand::thread_rng();
  let mut hashmap = HashMap::new();

  for k in lengths.iter() {
    let path = format!("{}/{}-{}.params", cqlin_params_dir, k, params.k());
    let params_path = Path::new(&path);
    if File::open(&params_path).is_err() {
      // attempt to resize the params
      let mut cqlin_params = params.clone();
      cqlin_params.resize(*k as u32);

      let mut buf = vec![];
      cqlin_params.write(&mut buf).expect("Failed to write cqlin params");
      let mut file = File::create(&params_path).expect("Failed to create cqlin params file");
      file
        .write_all(&buf[..])
        .expect("Failed to write cqlin params to file");

      let mut params_fs = File::open(&params_path).expect("couldn't load params");
      let params = ParamsKZG::<Bn256>::read(&mut params_fs).expect("Failed to read params");
      hashmap.insert(*k, params);
    }
  }

  hashmap
}

// pub fn get_cq_kzg_params(cq_params_dir: &str, )

pub fn serialize(data: &Vec<u8>, path: &str) -> u64 {
  let mut file = File::create(path).unwrap();
  file.write_all(data).unwrap();
  file.metadata().unwrap().len()
}

pub fn verify_kzg(
  params: &ParamsKZG<Bn256>,
  cqlin_params: &HashMap<usize, &ParamsKZG<Bn256>>,
  vk: &VerifyingKey<Bn256, G1Affine>,
  strategy: SingleStrategy<Bn256>,
  mut small_strategy: HashMap<usize, SingleStrategy<Bn256>>,
  public_vals: &Vec<Fr>,
  mut transcript: Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
) {
  assert!(
    verify_proof::<
      Bn256,
      KZGCommitmentScheme<Bn256>,
      VerifierSHPLONK<'_, Bn256>,
      Challenge255<G1Affine>,
      Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
    >(
      &params,
      cqlin_params,
      &vk,
      strategy,
      small_strategy,
      &[&[&public_vals]], 
      &mut transcript
    )
    .is_ok(),
    "proof did not verify"
  );
}

pub fn time_circuit_kzg(circuit: ModelCircuit<Fr>) {
  let rng = rand::thread_rng();
  let start = Instant::now();

  // ZKML-TODO: Ensure that this is accurate;
  let degree = circuit.k as u32;
  let cqlin_degree = circuit.k as u32;
  let cq_degree = circuit.k as u32;
  let params = get_kzg_params("./params_kzg", degree, cqlin_degree, cq_degree);
  let cqlin_params = get_cqlin_kzg_params(&params, "./cqlin_params_kzg", vec![]);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );

  let vk_circuit = circuit.clone();
  let vk = keygen_vk(&params, &cqlin_params, &vk_circuit).unwrap();
  drop(vk_circuit);
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let vkey_size = serialize(&vk.to_bytes(SerdeFormat::RawBytes), "vkey");
  println!("vkey size: {} bytes", vkey_size);

  let pk_circuit = circuit.clone();
  let pk = keygen_pk(&params, vk, &pk_circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  drop(pk_circuit);

  let pkey_size = serialize(&pk.to_bytes(SerdeFormat::RawBytes), "pkey");
  println!("pkey size: {} bytes", pkey_size);

  let fill_duration = start.elapsed();
  let proof_circuit = circuit.clone();
  let _prover = MockProver::run(degree, &proof_circuit, vec![vec![]]).unwrap();
  let public_vals = get_public_values();
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  // Convert public vals to serializable format
  let public_vals_u8: Vec<u8> = public_vals
    .iter()
    .map(|v: &Fr| v.to_bytes().to_vec())
    .flatten()
    .collect();
  let public_vals_u8_size = serialize(&public_vals_u8, "public_vals");
  println!("Public vals size: {} bytes", public_vals_u8_size);

  let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
  create_proof::<
    Bn256,
    KZGCommitmentScheme<Bn256>,
    ProverSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    _,
    Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
    ModelCircuit<Fr>,
  >(
    &params,
    &cqlin_params,
    &pk,
    &[proof_circuit],
    &[&[&public_vals]],
    rng,
    &mut transcript,
  )
  .unwrap();
  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  println!("Proving time: {:?}", proof_duration - fill_duration);

  let proof_size = serialize(&proof, "proof");
  let proof = std::fs::read("proof").unwrap();

  println!("Proof size: {} bytes", proof_size);

  let strategy = SingleStrategy::new(&params);
  let mut small_strategies = HashMap::new();
  for (length, param) in cqlin_params.iter(){ 
    small_strategies.insert(*length, SingleStrategy::new(&params));
  }

  let cqlin_params_rf: HashMap<_, _> = cqlin_params
    .iter()
    .map(|(&len, params)| (len, params))
    .collect();

  let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

  println!("public vals: {:?}", public_vals);
  verify_kzg(
    &params,
    &cqlin_params_rf,
    &pk.get_vk(),
    strategy,
    small_strategies,
    &public_vals,
    transcript_read,
  );
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);
}

// Standalone verification
pub fn verify_circuit_kzg(
  circuit: ModelCircuit<Fr>,
  vkey_fname: &str,
  proof_fname: &str,
  public_vals_fname: &str,
) {
  // ZKML TODO
  let degree = circuit.k as u32;
  let cqlin_degree = circuit.k as u32;
  let cq_degree = circuit.k as u32;

  let params = get_kzg_params("./params_kzg", degree, cqlin_degree, cq_degree);
  let cqlin_params = get_cqlin_kzg_params(&params, "./cqlin_params_kzg", vec![]);

  println!("Loaded the parameters");

  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open(vkey_fname).unwrap()),
    SerdeFormat::RawBytes
  )
  .unwrap();
  println!("Loaded vkey");

  let proof = std::fs::read(proof_fname).unwrap();

  let public_vals_u8 = std::fs::read(&public_vals_fname).unwrap();
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|chunk| Fr::from_bytes(chunk.try_into().expect("conversion failed")).unwrap())
    .collect();

  let strategy = SingleStrategy::new(&params);
  let mut small_strategies = HashMap::new();
  for (length, param) in cqlin_params.iter(){ 
    small_strategies.insert(*length, SingleStrategy::new(&params));
  }

  let cqlin_params_rf: HashMap<_, _> = cqlin_params
    .iter()
    .map(|(&len, params)| (len, params))
    .collect();

  let transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

  let start = Instant::now();
  let verify_start = start.elapsed();
  verify_kzg(&params, &cqlin_params_rf, &vk, strategy, small_strategies, &public_vals, transcript);
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - verify_start);
  println!("Proof verified!")
}
