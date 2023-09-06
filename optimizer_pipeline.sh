d=20
c=10
commitment="ipa"
cd examples/benchmark
python keras_create_models.py ${d}
python keras_to_tflite.py
python ../../python/create_logical.py --model model.tflite --model_output_dir ./ --config_output ./ --scale_factor 512 --k 0 --num_cols ${c} --num_randoms 1024

n=$(ls | egrep "config_d*" | wc -l)
n=$(( $n - 1 ))
break
for i in $( seq 0 $n )
do
python ../../python/input_converter.py --model_config model_${i}.msgpack --inputs 7.npy --output example_inp.msgpack
cd ../..
./target/release/estimate_cost examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack ${commitment}
cd examples/benchmark
done
cd ../..
python python/update_msgpack.py --model_input examples/benchmark/converted_model.msgpack --config_output examples/benchmark/config.msgpack
./target/release/time_circuit examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack ${commitment}

#cd ../..
#./target/release/estimate_size examples/benchmark/converted_model.msgpack
#./target/release/estimate_cost examples/benchmark/converted_model.msgpack
#./target/release/test_circuit examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack
#./target/release/time_circuit examples/benchmark/converted_model.msgpack examples/benchmark/example_inp.msgpack kzg