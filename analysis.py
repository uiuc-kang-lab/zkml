import json
import msgpack
import os

msg_dict = dict()
for bench in ["ipa_fft", "ipa_msm", "ipa_permute", "kzg_fft", "kzg_msm"]:
    path = os.path.join("target/criterion", bench, "k")
    sub_dirs = os.listdir(path)
    sub_dirs.remove("report")
    sub_dirs.sort()
    current_dict = dict()
    for sub_dir in sub_dirs:
        with open(os.path.join(path, sub_dir, "new/estimates.json"), "r") as f:
            data = json.load(f)
            current_dict[sub_dir] = data["mean"]["point_estimate"]
    msg_dict[bench] = current_dict

for bench in ["ipa_mul", "kzg_mul"]:
    path = os.path.join("target/criterion", bench, "/new/estimates.json")
    with open(path, "r") as f:
        data = json.load(f)
        msg_dict[bench] = data["mean"]["point_estimate"]
print(msg_dict)