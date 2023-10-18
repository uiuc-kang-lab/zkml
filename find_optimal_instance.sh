commitment=$1
working_dir=$2
i=$3
c=$4

cp ${working_dir}/model_${i}.msgpack ${working_dir}/tmp_${i}_${c}.msgpack

python python/update_msgpack.py --model_input ${working_dir}/tmp_${i}_${c}.msgpack --c ${c}
python python/fake_input_converter.py --model_config ${working_dir}/tmp_${i}_${c}.msgpack --output ${working_dir}/example_inp_${i}_${c}.msgpack
output=$(./target/release/estimate_cost ${working_dir}/tmp_${i}_${c}.msgpack ${working_dir}/example_inp_${i}_${c}.msgpack ${commitment})
if [[ $output =~ Total\ time\ cost\ \(esitmated\):\ ([0-9]+\.[0-9]+) ]]; then
  estimated_time_ns=${BASH_REMATCH[1]}
  estimated_time_sec=$(echo "scale=3; $estimated_time_ns / 1000000000" | bc)
fi
if [[ $output =~ Total\ number\ of\ rows:\ ([0-9]+) ]]; then
  estimated_row=${BASH_REMATCH[1]}
fi
if [[ $output =~ Optimal\ k:\ ([0-9]+) ]]; then
  estimated_k=${BASH_REMATCH[1]}
fi
rm ${working_dir}/example_inp_${i}_${c}.msgpack
rm ${working_dir}/tmp_${i}_${c}.msgpack
echo "### estimated_time=$estimated_time_sec | row=$estimated_row | k=$estimated_k | c=$c | i=$i"