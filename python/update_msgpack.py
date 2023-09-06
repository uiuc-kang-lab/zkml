import msgpack
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_input", type=str, required=True)
parser.add_argument("--config_input", type=str, required=True)
args = parser.parse_args()

with open("k.tmp", "r+") as data_file:
    k = int(data_file.readline())

with open(args.model_input, "rb") as data_file:
    byte_data = data_file.read()

data_loaded = msgpack.unpackb(byte_data)
data_loaded['k'] = k
with open(args.model_input, "wb") as outfile:
    packed = msgpack.packb(data_loaded, use_bin_type=True)
    outfile.write(packed)

data_loaded['tensors'] = []
with open(args.config_input, "wb") as outfile:
    config_packed = msgpack.packb(data_loaded, use_bin_type=True)
    outfile.write(config_packed)


