# Install Rust stuff
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup override set nightly

# Install Gdown
pip install gdown
mkdir examples/benchmark
gdown https://drive.google.com/drive/folders/1OTMmxUKR8hzWTSOZlj3JBgWSuc3RKF8x -O ./examples/benchmark --folder

# Cargo build
cargo build --release
./target/release/time_circuit examples/benchmark/converted_model_20.msgpack examples/benchmark/example_inp_20.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_40.msgpack examples/benchmark/example_inp_40.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_80.msgpack examples/benchmark/example_inp_80.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_160.msgpack examples/benchmark/example_inp_160.msgpack ipa >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_20.msgpack examples/benchmark/example_inp_20.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_40.msgpack examples/benchmark/example_inp_40.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_80.msgpack examples/benchmark/example_inp_80.msgpack kzg >> experiment.txt
./target/release/time_circuit examples/benchmark/converted_model_160.msgpack examples/benchmark/example_inp_160.msgpack kzg >> experiment.txt

curl -d "AWS: benchmark models successfully complete" ntfy.sh/bjchen4