# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
./target/release/time_circuit examples/benchmark/converted_model_20.msgpack examples/benchmark/example_inp_20.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_40.msgpack examples/benchmark/example_inp_40.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_80.msgpack examples/benchmark/example_inp_80.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_160.msgpack examples/benchmark/example_inp_160.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_20.msgpack examples/benchmark/example_inp_20.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_40.msgpack examples/benchmark/example_inp_40.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_80.msgpack examples/benchmark/example_inp_80.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_160.msgpack examples/benchmark/example_inp_160.msgpack kzg >> experiment.txt

curl -d "AWS: benchmark models successfully complete" ntfy.sh/bjchen4