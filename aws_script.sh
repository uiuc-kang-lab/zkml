# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
./target/release/time_circuit testing/data/cifar10/model_0.msgpack testing/data/cifar10/example_inp.msgpack ipa > debug_cifar10_ipa.txt
./target/release/time_circuit testing/data/cifar10/model_0.msgpack testing/data/cifar10/example_inp.msgpack kzg > debug_cifar10_kzg.txt
./target/release/time_circuit testing/data/gpt2/model.msgpack testing/data/gpt2/example_inp.msgpack ipa > debug_gpt2_ipa.txt
./target/release/time_circuit testing/data/gpt2/model.msgpack testing/data/gpt2/example_inp.msgpack kzg > debug_gpt2_kzg.txt
./target/release/time_circuit testing/data/mobilenet_1.0/model_0.msgpack testing/data/mobilenet_1.0/example_inp.msgpack ipa > debug_mobilenet_1.0_ipa.txt
./target/release/time_circuit testing/data/mobilenet_1.0/model_0.msgpack testing/data/mobilenet_1.0/example_inp.msgpack kzg > debug_mobilenet_1.0_kzg.txt
./target/release/time_circuit testing/data/mobilenet_1.4/model_0.msgpack testing/data/mobilenet_1.4/example_inp.msgpack ipa > debug_mobilenet_1.4_ipa.txt
./target/release/time_circuit testing/data/mobilenet_1.4/model_0.msgpack testing/data/mobilenet_1.4/example_inp.msgpack kzg > debug_mobilenet_1.4_kzg.txt
./target/release/time_circuit testing/data/twitter2/model_0.msgpack testing/data/twitter2/example_inp.msgpack ipa > debug_twitter2_ipa.txt
./target/release/time_circuit testing/data/twitter2/model_0.msgpack testing/data/twitter2/example_inp.msgpack kzg > debug_twitter2_kzg.txt

curl -d "AWS: benchmark models successfully complete" ntfy.sh/bjchen4