d=20
k=18
c=10
commitment="kzg"
cd examples/benchmark
python keras_create_models.py ${d}
python keras_to_tflite.py
python ../../python/converter.py --model model.tflite --model_output converted_model.msgpack --config_output config.msgpack --scale_factor 512 --k ${k} --num_cols ${c} --num_randoms 2048
python ../../python/input_converter.py --model_config converted_model.msgpack --inputs 7.npy --output example_inp.msgpack
cd ../..
./target/release/estimate_cost examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack ${commitment}
#python python/update_msgpack.py --model_input examples/benchmark/converted_model.msgpack --config_input examples/benchmark/config.msgpack
#./target/release/time_circuit examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack ${commitment}