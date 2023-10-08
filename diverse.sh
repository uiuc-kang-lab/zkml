for i in $( seq 3 5)
do
bash find_optimal.sh kzg examples/diverse/${i} 1024 >> examples/diverse/${i}/bash.log
bash find_optimal.sh ipa examples/diverse/${i} 1024 >> examples/diverse/${i}/bash.log
done