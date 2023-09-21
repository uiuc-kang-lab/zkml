# kzg examples/benchmark 1024
commitment=$1
working_dir=$2
num_randoms=$3

best_msgpack=""
best_time=9999999999999999999999999999.99
best_c=0

python python/create_logical.py --model ${working_dir}/model.tflite --model_output_dir ${working_dir} --config_output_dir ${working_dir} --scale_factor 512 --num_randoms ${num_randoms}

n=$(ls ${working_dir} | egrep "config_d*" | wc -l)
n=$(( $n - 1 ))
echo "n=$n"
for c in $(seq 60 10 70)
do
  for i in $( seq 0 $n )
  do
    python python/update_msgpack.py --model_input ${working_dir}/model_${i}.msgpack --config_input ${working_dir}/config_${i}.msgpack --c ${c}
    python python/fake_input_converter.py --model_config ${working_dir}/model_${i}.msgpack --output ${working_dir}/example_inp.msgpack
    output=$(./target/release/estimate_cost ${working_dir}/model_${i}.msgpack ${working_dir}/example_inp.msgpack ${commitment})
    echo "output=$output"
    if [[ $output =~ Total\ time\ cost\ \(esitmated\):\ ([0-9]+\.[0-9]+) ]]; then
      estimated_time_ns=${BASH_REMATCH[1]}
      estimated_time_sec=$(echo "scale=2; $estimated_time_ns / 1000000000" | bc)
      echo "estimated_time_sec=$estimated_time_sec"
      if [ $(echo "$estimated_time_sec < $best_time" | bc) -eq 1 ]; then
        best_msgpack=${working_dir}/model_${i}.msgpack
        best_time=$estimated_time_sec
        best_c=$c
      fi
    fi
    if [[ $output =~ Optimal\ k:\ ([0-9]+) ]]; then
      estimated_k=${BASH_REMATCH[1]}
      echo "estimated_k=$estimated_k"
      python python/update_msgpack.py --model_input ${working_dir}/model_${i}.msgpack --config_input ${working_dir}/config_${i}.msgpack --k ${estimated_k}
    fi
    # Real Proving with Halo2...
    #for j in $( seq 0 1 )
    #do
    #    output=$(./target/release/time_circuit ${working_dir}/model_${i}.msgpack ${working_dir}/example_inp.msgpack ${commitment})
    #    if [[ $output =~ Proving\ time:\ ([0-9]+\.[0-9]+) ]]; then
    #        proving_time=${BASH_REMATCH[1]}
    #        proving_time_sec=$(echo "scale=2; $proving_time / 1" | bc)
    #    fi
    #    echo "$commitment $i $c $proving_time_sec" >> my_exp.txt
    #done
  done
done
echo "Best msgpack: $best_msgpack when c=$best_c"
echo "Its estimated proving time: $best_time"
