import msgpack
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_input", type=str, required=True)
  parser.add_argument("--config_input", type=str, default=None)
  parser.add_argument("--c", type=int, default=-1)
  parser.add_argument("--k", type=int, default=-1)
  args = parser.parse_args()
  k = args.k
  c = args.c

  updated = False
  with open(args.model_input, "rb") as data_file:
      byte_data = data_file.read()
      data_loaded = msgpack.unpackb(byte_data)
  
  if k != -1:
    data_loaded['k'] = k
    updated = True
  if c != -1:
    data_loaded['num_cols'] = c
    updated = True

  if updated:
    with open(args.model_input, "wb") as outfile:
        packed = msgpack.packb(data_loaded, use_bin_type=True)
        outfile.write(packed)
    
    data_loaded['tensors'] = []
    if args.config_input is not None:
      with open(args.config_input, "wb") as outfile:
          config_packed = msgpack.packb(data_loaded, use_bin_type=True)
          outfile.write(config_packed)

if __name__ == '__main__':
  main()