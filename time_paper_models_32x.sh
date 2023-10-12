# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
export INFO=1
export MEASURE=1

# ipa

# mobilenet_1.4 best: estimated_time=2971.040 | row=16641751 | k=24 | c=15 | i=0
#./target/release/time_circuit data/mobilenet_1.4/model_best.msgpack data/mobilenet_1.4/example_inp.msgpack ipa > data/mobilenet_1.4/model_best_ipa.txt
#rm pkey
# mobilenet_1.4 40: estimated_time=8760.331 | row=9142025 | k=24 | c=40 | i=0
#./target/release/time_circuit data/mobilenet_1.4/model_40.msgpack data/mobilenet_1.4/example_inp.msgpack ipa > data/mobilenet_1.4/model_40_ipa.txt
#rm pkey

# gpt best: | k=24 | c=25 | i=0
./target/release/time_circuit data/gpt2/model_best_ipa.msgpack data/gpt2/inp.msgpack ipa > data/gpt2/model_best_ipa.txt
rm pkey
# gpt 40: | k=24 | c=40 | i=0
./target/release/time_circuit data/gpt2/model_40.msgpack data/gpt2/inp.msgpack ipa > data/gpt2/model_40_ipa.txt
rm pkey

# kzg

# mobilenet_1.4 best: estimated_time=2971.040 | row=16641751 | k=24 | c=15 | i=0
#./target/release/time_circuit data/mobilenet_1.4/model_best.msgpack data/mobilenet_1.4/example_inp.msgpack kzg > data/mobilenet_1.4/model_best_kzg.txt
#rm pkey
# mobilenet_1.4 40: estimated_time=8760.331 | row=9142025 | k=24 | c=40 | i=0
#./target/release/time_circuit data/mobilenet_1.4/model_40.msgpack data/mobilenet_1.4/example_inp.msgpack kzg > data/mobilenet_1.4/model_40_kzg.txt
#rm pkey

# gpt best: | k=24 | c=25 | i=0
./target/release/time_circuit data/gpt2/model_best_kzg.msgpack data/gpt2/inp.msgpack kzg > data/gpt2/model_best_kzg.txt
rm pkey
# gpt 40: | k=24 | c=40 | i=0
./target/release/time_circuit data/gpt2/model_40.msgpack data/gpt2/inp.msgpack kzg > data/gpt2/model_40_kzg.txt
rm pkey
