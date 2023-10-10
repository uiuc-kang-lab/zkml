# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
export INFO=1
export MEASURE=1

# ipa

# cifar10 best: estimated_time=171.110 | row=481247 | k=19 | c=15 | i=0
#./target/release/time_circuit data/cifar10/model_best.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_best_ipa.txt
#rm pkey
# cifar10 40: estimated_time=218.843 | row=189828 | k=18 | c=40 | i=0
#./target/release/time_circuit data/cifar10/model_40.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_40_ipa.txt
#rm pkey
# cifar10 best diff: estimated_time=2373.294 | row=6741180 | k=23 | c=15 | i=6
#./target/release/time_circuit data/cifar10/model_best_diff.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_best_diff_ipa.txt
#rm pkey

# mobilenet_1.0 best: estimated_time=2909.653 | row=8104723 | k=23 | c=20 | i=0
#./target/release/time_circuit data/mobilenet_1.0/model_best.msgpack data/mobilenet_1.0/example_inp.msgpack ipa > data/mobilenet_1.0/model_best_ipa.txt
#rm pkey
# mobilenet_1.0 40: estimated_time=5927.144 | row=6372465 | k=23 | c=40 | i=0
./target/release/time_circuit data/mobilenet_1.0/model_40.msgpack data/mobilenet_1.0/example_inp.msgpack ipa > data/mobilenet_1.0/model_40_ipa.txt
rm pkey

# twitter2 best: estimated_time=477.073 | row=4109452 | k=22 | c=15 | i=0
#./target/release/time_circuit data/twitter2/model_best.msgpack data/twitter2/example_inp.msgpack ipa > data/twitter2/model_best_ipa.txt
#rm pkey
# twitter2 40: estimated_time=668.196 | row=1533671 | k=21 | c=40 | i=0
#./target/release/time_circuit data/twitter2/model_40.msgpack data/twitter2/example_inp.msgpack ipa > data/twitter2/model_40_ipa.txt
#rm pkey

# kzg

# cifar10 best: estimated_time=171.110 | row=481247 | k=19 | c=15 | i=0
#./target/release/time_circuit data/cifar10/model_best.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_best_kzg.txt
#rm pkey
# cifar10 40: estimated_time=218.843 | row=189828 | k=18 | c=40 | i=0
#./target/release/time_circuit data/cifar10/model_40.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_40_kzg.txt
#rm pkey
# cifar10 best diff: estimated_time=2373.294 | row=6741180 | k=23 | c=15 | i=6
#./target/release/time_circuit data/cifar10/model_best_diff.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_best_diff_kzg.txt
#rm pkey

# mobilenet_1.0 best: estimated_time=2909.653 | row=8104723 | k=23 | c=20 | i=0
#./target/release/time_circuit data/mobilenet_1.0/model_best.msgpack data/mobilenet_1.0/example_inp.msgpack kzg > data/mobilenet_1.0/model_best_kzg.txt
#rm pkey
# mobilenet_1.0 40: estimated_time=5927.144 | row=6372465 | k=23 | c=40 | i=0
./target/release/time_circuit data/mobilenet_1.0/model_40.msgpack data/mobilenet_1.0/example_inp.msgpack kzg > data/mobilenet_1.0/model_40_kzg.txt
rm pkey

# twitter2 best: estimated_time=477.073 | row=4109452 | k=22 | c=15 | i=0
#./target/release/time_circuit data/twitter2/model_best.msgpack data/twitter2/example_inp.msgpack kzg > data/twitter2/model_best_kzg.txt
#rm pkey
# twitter2 40: estimated_time=668.196 | row=1533671 | k=21 | c=40 | i=0
#./target/release/time_circuit data/twitter2/model_40.msgpack data/twitter2/example_inp.msgpack kzg > data/twitter2/model_40_kzg.txt
#rm pkey

curl -d "AWS: benchmark paper models successfully complete" ntfy.sh/bjchen4