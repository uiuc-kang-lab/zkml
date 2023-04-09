use std::{
  fs::File,
  io::{BufReader, Cursor, Read, Write},
  path::Path,
  time::Instant,
};

use halo2_proofs::{
  dev::MockProver,
  halo2curves::bn256::{Bn256, Fr, G1Affine},
  plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, VerifyingKey},
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

use super::loader::ModelMsgpack;

pub fn get_kzg_params(params_dir: &str, degree: u32) -> ParamsKZG<Bn256> {
  let rng = rand::thread_rng();
  let path = format!("{}/{}.params", params_dir, degree);
  let params_path = Path::new(&path);
  if File::open(&params_path).is_err() {
    let params = ParamsKZG::<Bn256>::setup(degree, rng);
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

pub fn serialize(data: &Vec<u8>, path: &str) -> u64 {
  let mut file = File::create(path).unwrap();
  file.write_all(data).unwrap();
  file.metadata().unwrap().len()
}

pub fn verify_kzg(
  params: &ParamsKZG<Bn256>,
  vk: &VerifyingKey<G1Affine>,
  strategy: SingleStrategy<Bn256>,
  public_vals: &Vec<Fr>,
  mut transcript: Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
) {
  assert!(
    verify_proof::<
      KZGCommitmentScheme<Bn256>,
      VerifierSHPLONK<'_, Bn256>,
      Challenge255<G1Affine>,
      Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
    >(&params, &vk, strategy, &[&[&public_vals]], &mut transcript)
    .is_ok(),
    "proof did not verify"
  );
}

pub fn time_circuit_kzg(circuit: ModelCircuit<Fr>, config: ModelMsgpack) {
  let rng = rand::thread_rng();
  let start = Instant::now();

  let empty_circuit = circuit.clone();
  let proof_circuit = circuit.clone();

  let degree = config.k.try_into().unwrap();
  let params = get_kzg_params("./params_kzg", degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );

  let vk = keygen_vk(&params, &empty_circuit).unwrap();
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let vkey_size = serialize(&vk.to_bytes(SerdeFormat::RawBytes), "vkey");
  println!("vkey size: {} bytes", vkey_size);
  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open("vkey").unwrap()),
    SerdeFormat::RawBytes,
  )
  .unwrap();

  let pk = keygen_pk(&params, vk, &empty_circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  drop(empty_circuit);

  let pkey_size = serialize(&pk.to_bytes(SerdeFormat::RawBytes), "pkey");
  println!("pkey size: {} bytes", pkey_size);

  let fill_duration = start.elapsed();
  let _prover =
    MockProver::run(config.k.try_into().unwrap(), &proof_circuit, vec![vec![]]).unwrap();
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
    KZGCommitmentScheme<Bn256>,
    ProverSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    _,
    Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
    ModelCircuit<Fr>,
  >(
    &params,
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
  let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  println!("transcript: {:?}", transcript_read);
  println!("proof: {:?}", proof[0..10].to_vec());
  println!("proof len: {}", proof.len());
  println!("vk: {:?}", pk.get_vk().fixed_commitments());

  println!("public vals: {:?}", public_vals);
  verify_kzg(
    &params,
    &pk.get_vk(),
    strategy,
    &public_vals,
    transcript_read,
  );
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);

  // Prove again
  let rng = rand::thread_rng();
  let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
  create_proof::<
    KZGCommitmentScheme<Bn256>,
    ProverSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    _,
    Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
    ModelCircuit<Fr>,
  >(
    &params,
    &pk,
    &[circuit],
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
  let transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  println!("transcript: {:?}", transcript);
  println!("proof: {:?}", proof[0..10].to_vec());
  println!("proof len: {}", proof.len());
  println!("vk: {:?}", pk.get_vk().fixed_commitments());
  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open("vkey").unwrap()),
    SerdeFormat::RawBytes,
  )
  .unwrap();

  println!("public vals: {:?}", public_vals);
  verify_kzg(&params, &vk, strategy, &public_vals, transcript);
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);
}

pub fn verify_circuit_kzg(config: ModelMsgpack) {
  let degree = config.k.try_into().unwrap();
  let params = get_kzg_params("./params_kzg", degree);

  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open("vkey").unwrap()),
    SerdeFormat::RawBytes,
  )
  .unwrap();
  println!("vk: {:?}", vk.fixed_commitments());

  let vkey_size = serialize(&vk.to_bytes(SerdeFormat::RawBytes), "vkey");
  println!("vkey size: {} bytes", vkey_size);

  let proof = std::fs::read("proof").unwrap();

  let public_vals_u8 = std::fs::read("public_vals").unwrap();
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|chunk| Fr::from_bytes(chunk.try_into().expect("conversion failed")).unwrap())
    .collect();

  let strategy = SingleStrategy::new(&params);
  let tmp = proof.clone();
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&tmp[..]);
  println!("transcript: {:?}", transcript);
  let proof_sum = proof.iter().fold(0_u64, |acc, x| acc + *x as u64);
  println!("proof sum: {}", proof_sum);
  println!("proof: {:?}", proof[0..10].to_vec());
  println!("proof len: {}", proof.len());

  println!("public vals: {:?}", public_vals);
  /*
  if let Err(e) = verify_proof::<
    KZGCommitmentScheme<Bn256>,
    VerifierSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
    halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
  >(
    &params,
    &vk,
    strategy.clone(),
    &[&[&public_vals]],
    &mut transcript,
  ) {
    println!("proof did not verify: {:?}", e);
  }
  */

  verify_kzg(&params, &vk, strategy, &public_vals, transcript)
}

pub fn verify_circuit_kzg_tmp(
  config: ModelMsgpack,
  model_fname: &str,
  input_fname: &str,
  vkey_path: &str,
  proof_path: &str,
  public_vals_path: &str,
) {
  let degree = config.k.try_into().unwrap();
  let params = get_kzg_params("./params_kzg", degree);

  let vkey_fs = File::open(vkey_path).expect("couldn't load vkey");
  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(vkey_fs),
    SerdeFormat::RawBytes,
  )
  .unwrap();

  let mut proof_fs = File::open(proof_path).expect("couldn't load proof");
  let mut proof = Vec::new();
  proof_fs.read_to_end(&mut proof).unwrap();
  println!("proof size: {} bytes", proof.len());
  let proof_sum = proof.iter().fold(0_u64, |acc, x| acc + *x as u64);
  println!("proof sum: {}", proof_sum);

  let strategy = SingleStrategy::new(&params);
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

  let public_vals_u8 = std::fs::read(public_vals_path).unwrap();
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|chunk| Fr::from_bytes(chunk.try_into().expect("conversion failed")).unwrap())
    .collect();
  println!("public vals: {:?}", public_vals);
  assert!(
    verify_proof::<
      KZGCommitmentScheme<Bn256>,
      VerifierSHPLONK<'_, Bn256>,
      Challenge255<G1Affine>,
      Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
    >(&params, &vk, strategy, &[&[&public_vals]], &mut transcript)
    .is_ok(),
    "proof did not verify"
  );
}
