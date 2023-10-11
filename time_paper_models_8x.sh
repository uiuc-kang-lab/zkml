# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
export INFO=1
export MEASURE=1

# ipa

# cifar10 best: estimated_time=71.474 | row=260851 | k=18 | c=27 | i=0
./target/release/time_circuit data/cifar10/model_best.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_best_ipa.txt
rm pkey
# cifar10 40: estimated_time=110.920 | row=182923 | k=18 | c=40 | i=0
./target/release/time_circuit data/cifar10/model_40.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_40_ipa.txt
rm pkey
# cifar10 best diff: estimated_time=1270.840 | row=3620156 | k=22 | c=27 | i=1
./target/release/time_circuit data/cifar10/model_best_diff.msgpack data/cifar10/example_inp.msgpack ipa > data/cifar10/model_best_diff_ipa.txt
rm pkey

# dlrm best: estimated_time=38.562 | row=131058 | k=17 | c=27 | i=0
./target/release/time_circuit data/dlrm/model_best.msgpack data/dlrm/example_inp.msgpack ipa > data/dlrm/model_best_ipa.txt
rm pkey
# dlrm 40: estimated_time=59.293 | row=91053 | k=17 | c=40 | i=0
./target/release/time_circuit data/dlrm/model_40.msgpack data/dlrm/example_inp.msgpack ipa > data/dlrm/model_40_ipa.txt
rm pkey
# dlrm best diff: estimated_time=1193.792 | row=2221303 | k=22 | c=27 | i=1
./target/release/time_circuit data/dlrm/model_best_diff.msgpack data/dlrm/example_inp.msgpack ipa > data/dlrm/model_best_diff_ipa.txt

# vgg16 best: estimated_time=737.912 | row=3731556 | k=22 | c=13 | i=0
./target/release/time_circuit data/vgg16/model_best.msgpack data/vgg16/example_inp.msgpack ipa > data/vgg16/model_best_ipa.txt
rm pkey
# vgg16 40: estimated_time=1178.605 | row=1930746 | k=21 | c=40 | i=0
./target/release/time_circuit data/vgg16/model_40.msgpack data/vgg16/example_inp.msgpack ipa > data/vgg16/model_40_ipa.txt
rm pkey
# vgg16 best diff: estimated_time=10267.363 | row=57767386 | k=26 | c=13 | i=1
./target/release/time_circuit data/vgg16/model_best_diff.msgpack data/vgg16/example_inp.msgpack ipa > data/vgg16/model_best_diff_ipa.txt
rm pkey

# twitter2 best: estimated_time=477.073 | row=4109452 | k=22 | c=15 | i=0
#./target/release/time_circuit data/twitter2/model_best.msgpack data/twitter2/example_inp.msgpack ipa > data/twitter2/model_best_ipa.txt
#rm pkey
# twitter2 40: estimated_time=668.196 | row=1533671 | k=21 | c=40 | i=0
#./target/release/time_circuit data/twitter2/model_40.msgpack data/twitter2/example_inp.msgpack ipa > data/twitter2/model_40_ipa.txt
#rm pkey

# kzg

# cifar10 best
./target/release/time_circuit data/cifar10/model_best.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_best_kzg.txt
rm pkey
# cifar10 40
./target/release/time_circuit data/cifar10/model_40.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_40_kzg.txt
rm pkey
# cifar10 best diff
./target/release/time_circuit data/cifar10/model_best_diff.msgpack data/cifar10/example_inp.msgpack kzg > data/cifar10/model_best_diff_kzg.txt
rm pkey

# dlrm best
./target/release/time_circuit data/dlrm/model_best.msgpack data/dlrm/example_inp.msgpack kzg > data/dlrm/model_best_kzg.txt
rm pkey
# dlrm 40
./target/release/time_circuit data/dlrm/model_40.msgpack data/dlrm/example_inp.msgpack kzg > data/dlrm/model_40_kzg.txt
rm pkey
# dlrm best diff
./target/release/time_circuit data/dlrm/model_best_diff.msgpack data/dlrm/example_inp.msgpack kzg > data/dlrm/model_best_diff_kzg.txt

# vgg16 best
./target/release/time_circuit data/vgg16/model_best.msgpack data/vgg16/example_inp.msgpack kzg > data/vgg16/model_best_kzg.txt
rm pkey
# vgg16 40
./target/release/time_circuit data/vgg16/model_40.msgpack data/vgg16/example_inp.msgpack kzg > data/vgg16/model_40_kzg.txt
rm pkey
# vgg16 best diff
./target/release/time_circuit data/vgg16/model_best_diff.msgpack data/vgg16/example_inp.msgpack kzg > data/vgg16/model_best_diff_kzg.txt
rm pkey

# mobilenet

## mobilenet_1.0 best: estimated_time=1237.242 | row=7921655 | k=23 | c=20 | i=0
#./target/release/time_circuit data/mobilenet_1.0/model_best.msgpack data/mobilenet_1.0/example_inp.msgpack ipa > data/mobilenet_1.0/model_best_ipa.txt
#rm pkey
## mobilenet_1.0 40: estimated_time=2483.899 | row=6280928 | k=23 | c=40 | i=0
#./target/release/time_circuit data/mobilenet_1.0/model_40.msgpack data/mobilenet_1.0/example_inp.msgpack ipa > data/mobilenet_1.0/model_40_ipa.txt
#rm pkey
#
## mobilenet_1.0 best
#./target/release/time_circuit data/mobilenet_1.0/model_best.msgpack data/mobilenet_1.0/example_inp.msgpack kzg > data/mobilenet_1.0/model_best_kzg.txt
#rm pkey
## mobilenet_1.0 40
#./target/release/time_circuit data/mobilenet_1.0/model_40.msgpack data/mobilenet_1.0/example_inp.msgpack kzg > data/mobilenet_1.0/model_40_kzg.txt
##rm pkey

# twitter2 best: estimated_time=477.073 | row=4109452 | k=22 | c=15 | i=0
#./target/release/time_circuit data/twitter2/model_best.msgpack data/twitter2/example_inp.msgpack kzg > data/twitter2/model_best_kzg.txt
#rm pkey
# twitter2 40: estimated_time=668.196 | row=1533671 | k=21 | c=40 | i=0
#./target/release/time_circuit data/twitter2/model_40.msgpack data/twitter2/example_inp.msgpack kzg > data/twitter2/model_40_kzg.txt
#rm pkey

curl -d "AWS: benchmark paper models successfully complete" ntfy.sh/bjchen4