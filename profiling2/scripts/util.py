import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from oneflow import profiler
from oneflow.profiler.events import KernelEvent, CustomEvent
import numpy as np
from collections import defaultdict, OrderedDict, namedtuple
import copy
import os
import csv
import traceback

ProfileResult = namedtuple("ProfileResult", ["op", "shape", "time", "valid", "count"])
# ProfileResult = namedtuple("ProfileResult", ["op", "time", "count", "shape", "args"])

WARMUP_NUM = int(os.getenv("ONEFLOW_PROFILE_WARMUP_NUM", 10))
RUN_NUM = int(os.getenv("ONEFLOW_PROFILE_RUN_NUM", 1000))
DEVICE = os.getenv("ONEFLOW_PROFILE_DEVICE", "cuda")
OUTPUT = os.getenv("ONEFLOW_PROFILE_CSV", "op_prof").replace(".csv", "")

_csv_filename = OUTPUT + ".csv"
_txt_filename = OUTPUT + ".txt"

# if "_miss_" not in _csv_filename:
#     _csv_filename = _csv_filename.replace(".csv", f".{timestamp}.csv")


class ProfilerBase:
    def __init__(self, op_list=None, tensor_generator=None) -> None:
        self.op_list = op_list
        self.tensor_generator = tensor_generator

    def run(self):
        self.profile_oplist(
            op_list=self.op_list, tensor_generator=self.tensor_generator
        )


def get_device():
    return flow.device(DEVICE)


def _write_result(result, table):
    if not hasattr(_write_result, "csv_writer"):
        csvfile = open(_csv_filename, "w", encoding="utf8")
        _write_result.csv_writer = csv.writer(csvfile)
        _write_result.csv_writer.writerow(ProfileResult._fields)
    if not hasattr(_write_result, "txt_writer"):
        _write_result.txt_writer = open(_txt_filename, "w", encoding="utf8")

    _write_result.csv_writer.writerow(result)
    _write_result.txt_writer.write(table)


def _run_flow(func_name, *args, **kwargs):
    if callable(func_name):
        func = func_name
    else:
        func = eval(f"flow.{func_name}")
    for _ in range(WARMUP_NUM):
        func(*args, **kwargs)
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("forward_total_time") as f:
            for _ in range(RUN_NUM):
                func(*args, **kwargs)
    prof.profile_events[:] = _filter_bad(prof.profile_events)
    return prof
    # with oneflow.profiler.record_function("lenet_backward_total_time") as f:
    #     eager_res.sum().backward()


def _filter_bad(events):
    stats = defaultdict(list)
    valid = []
    for e in events:
        if not e.has_cuda_time():
            continue
        key = e.key
        stats[key].append(copy.deepcopy(e))
    for key in stats:
        es = sorted(stats[key], key=lambda x: x.cuda_time_total)
        drop_head = 0.2
        drop_tail = 0.2
        b = int(len(es) * drop_head)
        e = int(len(es) * drop_tail)
        valid.extend(es[b:-e])
    return valid


def _get_oneflow_gpu_kernel_time(prof, raise_exp=False):
    kernel4count = list(
        filter(
            lambda x: x.name == "cudaLaunchKernel",
            copy.deepcopy(prof).key_averages(group_by_input_shape=True),
        )
    )[0]
    assert isinstance(kernel4count, CustomEvent)

    gpu_kernel_items = list(
        filter(
            lambda x: x.cuda_time_total is not None,
            copy.deepcopy(prof).key_averages(group_by_input_shape=True),
        )
    )

    # print(gpu_kernel_items)
    kernel_events = [i for i in gpu_kernel_items if isinstance(i, KernelEvent)]
    events_count = [i.count for i in gpu_kernel_items]

    # print(kernel_events, events_count, valid_count)
    event = kernel_events[0]

    assert len(kernel_events) == 1
    if len(set(events_count)) != 1:
        valid = kernel4count.count
    else:
        valid = events_count[0]

    assert valid > 200
    assert valid == max(events_count)

    kernel_gpu_time = sum(map(lambda x: x.cuda_time_total, gpu_kernel_items)) / valid

    result = ProfileResult(
        op=event.name,
        time=kernel_gpu_time,
        valid=valid,
        count=tuple(events_count),
        shape=event.input_shapes,
    )
    return result


def profile_op(func_name, *args, **kwargs):
    # print(func_name, str(args), str(kwargs))
    prof = _run_flow(func_name, *args, **kwargs)
    result = _get_oneflow_gpu_kernel_time(copy.deepcopy(prof))
    _write_result(result, prof.key_averages(group_by_input_shape=True).table())
    return result


def profile_oplist(op_list, tensor_generator):
    for op in op_list:
        for inputs in tensor_generator():
            if isinstance(inputs, tuple):
                result = profile_op(op, *inputs)
            else:
                result = profile_op(op, inputs)


def generate_one_tensor(logmax=30):
    for i in range(logmax):
        yield flow.randn(2**i, device=DEVICE)


def generate_two_tensor(logmax=30):
    for i in range(logmax):
        yield (flow.randn(2**i, device=DEVICE), flow.randn(2**i, device=DEVICE))
