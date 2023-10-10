# bash find_optimal.sh ipa testing/data/cifar10 1024 > exp/cifar10_ipa.txt
# bash find_optimal.sh kzg testing/data/cifar10 1024 > exp/cifar10_kzg.txt
# bash find_optimal.sh ipa testing/data/twitter2 1024 > exp/twitter2_ipa.txt
# bash find_optimal.sh kzg testing/data/twitter2 1024 > exp/twitter2_kzg.txt
# bash find_optimal.sh ipa testing/data/mobilenet_1.0 50000 > exp/mobilenet_1.0_ipa.txt
# bash find_optimal.sh kzg testing/data/mobilenet_1.0 50000 > exp/mobilenet_1.0_kzg.txt
bash find_optimal.sh ipa testing/data/mobilenet_1.4 50000 > exp/mobilenet_1.4_ipa.txt
bash find_optimal.sh kzg testing/data/mobilenet_1.4 50000 > exp/mobilenet_1.4_kzg.txt
bash find_optimal_gpt.sh ipa testing/data/gpt2 50000 > exp/gpt2_ipa.txt
bash find_optimal_gpt.sh kzg testing/data/gpt2 50000 > exp/gpt2_kzg.txt
curl -d "Estimate successful 😀" ntfy.sh/bjchen4
