#!/bin/bash
ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True \
nsys \
profile --stats=true \
--force-overwrite true \
-o tanh \
python3 -m oneflow.distributed.launch \
    --nproc_per_node 2 \
    scripts/profile_tanh.py --repeat 1000 --shape 536870912