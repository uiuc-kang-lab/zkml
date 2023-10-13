import json
import msgpack
import os
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", type=str, default="summary.msgpack")
    parser.add_argument("--show-plot", type=bool, default=False)
    return parser.parse_args()

def create_estimation_dict():
    msg_dict = dict()
    for bench in ["ipa_mul", "ipa_add", "ipa_fft", "ipa_msm", "ipa_permute", "kzg_mul", "kzg_add", "kzg_fft", "kzg_msm", "kzg_permute"]:
        path = os.path.join("target/criterion", bench, "k")
        sub_dirs = os.listdir(path)
        if "report" in sub_dirs:
            sub_dirs.remove("report")
        sub_dirs.sort()
        current_dict = dict()
        for sub_dir in sub_dirs:
            with open(os.path.join(path, sub_dir, "new/estimates.json"), "r") as f:
                data = json.load(f)
                current_dict[sub_dir] = data["mean"]["point_estimate"]
        msg_dict[bench] = current_dict

    return msg_dict

def main():
    args = parse_args()
    d = create_estimation_dict()

    summary_packed = msgpack.packb(d, use_bin_type=True)
    with open(args.summary_output, 'wb') as f:
        f.write(summary_packed)

    if args.show_plot:
        for bench in ["ipa_fft", "ipa_msm", "ipa_mul", "ipa_add", "ipa_permute", "kzg_fft", "kzg_msm", "kzg_mul", "kzg_add","kzg_permute"]:
            x = []
            y = []
            for k in d[bench]:
                if int(k)!=27:
                    x.append(int(k))
                    y.append(d[bench][k])
            plt.plot(x, y, label=bench)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()