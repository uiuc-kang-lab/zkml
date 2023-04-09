use halo2_proofs::halo2curves::bn256::Fr;
use zkml::{
  model::ModelCircuit,
  utils::{
    loader::{load_model_msgpack, ModelMsgpack},
    proving_kzg::verify_circuit_kzg,
  },
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let vkey_fname = std::env::args().nth(3).expect("verification key file path");
  let proof_fname = std::env::args().nth(4).expect("proof file path");
  let public_vals_path = std::env::args().nth(5).expect("public values file path");
  let kzg_or_ipa = std::env::args().nth(6).expect("kzg or ipa");

  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }

  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname);

  if kzg_or_ipa == "kzg" {
    verify_circuit_kzg(config);
  } else {
    // Serialization of the verification key doesn't seem to be supported for IPA
    panic!("Not implemented");
  }
}
