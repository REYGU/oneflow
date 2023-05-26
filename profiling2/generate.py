import glob
import os
import re

script_dir = "scripts"
files = glob.glob(os.path.join(script_dir, "profile_*.py"))
command = "CUDA_VISIBLE_DEVICES=4 ONEFLOW_PROFILE_CSV=results/profile_{name}  python3 scripts/profile_{name}.py"
output_path = "profiling.sh"

with open(output_path, "w", encoding="utf8") as fp:
    for file in files:
        name = re.search("profile_(\w+).py", file).group(1)
        fp.write(command.format(name=name) + "\n")
