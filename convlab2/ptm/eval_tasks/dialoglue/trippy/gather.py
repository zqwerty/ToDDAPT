import json
import sys
import os
import re

dir_prefix = sys.argv[1]
model_name = sys.argv[2]

eval_res = []

for seed in sys.argv[3:]:
    with open(os.path.join(dir_prefix, model_name, "seed{}".format(seed), "eval_pred_test.log"), "r") as f:
        line = f.readlines()[0]
        m = re.search(r"Joint goal acc: (0.\d+), .*", line)
        acc = float(m.group(1))
        eval_res.append(acc)

print(eval_res)
print("avg.:", sum(eval_res) / len(eval_res))

with open(os.path.join(dir_prefix, model_name, "all_eval_join_acc.txt"), "w") as f:
    f.write(str(eval_res) + "\n")
    f.write(str(sum(eval_res) / len(eval_res)) + "\n")

