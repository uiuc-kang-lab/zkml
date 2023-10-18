import subprocess
import argparse
from itertools import product

def main(args):

    # create logical implementations
    create_logical = "python python/create_logical.py --model {}/model.tflite --model_output_dir {} --config_output_dir {} --scale_factor 32 --num_randoms {}"
    subprocess.run(create_logical.format(args.working_dir, args.working_dir, args.working_dir, args.num_randoms), shell=True, text=True)

    # create physical implementations and find optimal instance
    template = 'bash find_optimal_instance.sh {} {} {} {}'

    commitment = args.commitment
    working_dir = args.working_dir

    best_est = 9999999999999999999999999999.99
    best_k, best_col, best_imp = 0, 0, 0

    # Run commands in parallel
    processes = []

    for i, c in product([i for i in range(args.max_i)], [c for c in range(args.min_c, args.max_c+1)]):
        command = template.format(commitment, working_dir, i, c)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)

    outputs = []
    for process in processes:
        stdout, stderr = process.communicate()
        outputs.append(stdout)

    for line in outputs:
        ### estimated_time=20.484 | row=5973 | k=13 | c=100 | i=0
        if '###' in line:
            est_time = float(line.split("|")[0].split("=")[1])
            k = int(line.split("|")[2].split("=")[1])
            col = int(line.split("|")[3].split("=")[1])
            imp = int(line.split("|")[4].split("=")[1])
            if est_time < best_est:
                best_est = est_time
                best_k = k
                best_col = col
                best_imp = imp
    print(f"Best estimated time: {best_est} | Best k: {best_k} | Best col: {best_col} | Best imp: {best_imp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commitment', type=str, default='kzg')
    parser.add_argument('--working_dir', type=str, default='examples/mnist')
    parser.add_argument('--max_i', type=int, default=1)
    parser.add_argument('--min_c', type=int, default=10)
    parser.add_argument('--max_c', type=int, default=100)
    parser.add_argument('--num_randoms', type=int, default=1024)
    args = parser.parse_args()
    main(args)
