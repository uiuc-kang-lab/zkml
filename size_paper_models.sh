# Cargo build
cargo build --release
mkdir params_ipa
mkdir params_kzg
export INFO=1
export MEASURE=1

# mnist: k=15, c=10
# dlrm: k=19, c=10
# resnet: k=20, c=10
# twitter: k=23, c=10
# vgg: k=23, c=10
# mobilenet: k=24, c=10

./target/release/time_circuit data_size/mnist/model_best.msgpack data_size/mnist/example_inp.msgpack kzg > data_size/mnist/model_best_kzg.txt
./target/release/time_circuit data_size/dlrm/model_best.msgpack data_size/dlrm/example_inp.msgpack kzg > data_size/dlrm/model_best_kzg.txt
./target/release/time_circuit data_size/cifar10/model_best.msgpack data_size/cifar10/example_inp.msgpack kzg > data_size/cifar10/model_best_kzg.txt
./target/release/time_circuit data_size/mnist/model_best.msgpack data_size/mnist/example_inp.msgpack ipa > data_size/mnist/model_best_ipa.txt
./target/release/time_circuit data_size/dlrm/model_best.msgpack data_size/dlrm/example_inp.msgpack ipa > data_size/dlrm/model_best_ipa.txt
./target/release/time_circuit data_size/cifar10/model_best.msgpack data_size/cifar10/example_inp.msgpack ipa > data_size/cifar10/model_best_ipa.txt
curl -d "AWS: benchmark MNIST, DLRM, CIFAR10 successfully complete" ntfy.sh/bjchen4
./target/release/time_circuit data_size/twitter2/model_best.msgpack data_size/twitter2/example_inp.msgpack kzg > data_size/twitter2/model_best_kzg.txt
./target/release/time_circuit data_size/vgg16/model_best.msgpack data_size/vgg16/example_inp.msgpack kzg > data_size/vgg16/model_best_kzg.txt
./target/release/time_circuit data_size/twitter2/model_best.msgpack data_size/twitter2/example_inp.msgpack ipa > data_size/twitter2/model_best_ipa.txt
./target/release/time_circuit data_size/vgg16/model_best.msgpack data_size/vgg16/example_inp.msgpack ipa > data_size/vgg16/model_best_ipa.txt
curl -d "AWS: benchmark Twitter, VGG16 successfully complete" ntfy.sh/bjchen4
./target/release/time_circuit data_size/mobilenet_1.0/model_best.msgpack data_size/mobilenet_1.0/example_inp.msgpack kzg > data_size/mobilenet_1.0/model_best_kzg.txt
./target/release/time_circuit data_size/mobilenet_1.0/model_best.msgpack data_size/mobilenet_1.0/example_inp.msgpack ipa > data_size/mobilenet_1.0/model_best_ipa.txt
curl -d "AWS: benchmark Mobilenet_1.0 successfully complete" ntfy.sh/bjchen4
curl -d "AWS: benchmark size models successfully complete" ntfy.sh/bjchen4