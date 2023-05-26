#!/bin/bash
ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True \
nsys \
profile --stats=true \
--force-overwrite true \
-o relu \
python3 -m oneflow.distributed.launch \
    --nproc_per_node 2 \
    profile_relu.py --repeat 10