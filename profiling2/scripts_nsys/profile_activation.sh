CUDA_VISIBLE_DEVICES=5,6    ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True nsys profile --stats=true --force-overwrite true -o results/1n1d_1000c/activation/134217728 python3  scripts/profile_activation.py --repeat 1000 --shape 134217728
python process_sqlite.py -s results/1n1d_1000c/activation/134217728.sqlite && rm results/1n1d_1000c/activation/134217728.qdrep && rm results/1n1d_1000c/activation/134217728.qdstrm && rm results/1n1d_1000c/activation/134217728.sqlite
CUDA_VISIBLE_DEVICES=5,6    ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True nsys profile --stats=true --force-overwrite true -o results/1n1d_1000c/activation/268435456 python3  scripts/profile_activation.py --repeat 1000 --shape 268435456
python process_sqlite.py -s results/1n1d_1000c/activation/268435456.sqlite && rm results/1n1d_1000c/activation/268435456.qdrep && rm results/1n1d_1000c/activation/268435456.qdstrm && rm results/1n1d_1000c/activation/268435456.sqlite
CUDA_VISIBLE_DEVICES=5,6    ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True nsys profile --stats=true --force-overwrite true -o results/1n1d_1000c/activation/536870912 python3  scripts/profile_activation.py --repeat 1000 --shape 536870912
python process_sqlite.py -s results/1n1d_1000c/activation/536870912.sqlite && rm results/1n1d_1000c/activation/536870912.qdrep && rm results/1n1d_1000c/activation/536870912.qdstrm && rm results/1n1d_1000c/activation/536870912.sqlite
CUDA_VISIBLE_DEVICES=5,6    ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True nsys profile --stats=true --force-overwrite true -o results/1n1d_1000c/activation/1073741824 python3  scripts/profile_activation.py --repeat 1000 --shape 1073741824
python process_sqlite.py -s results/1n1d_1000c/activation/1073741824.sqlite && rm results/1n1d_1000c/activation/1073741824.qdrep && rm results/1n1d_1000c/activation/1073741824.qdstrm && rm results/1n1d_1000c/activation/1073741824.sqlite
