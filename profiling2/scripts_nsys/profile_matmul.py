"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=int, default=100)
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--m", type=int, default=8)
parser.add_argument("--k", type=int, default=8)
args = parser.parse_args()


import oneflow as flow
from oneflow import nn
from oneflow.nn import functional as F
import os
import numpy as np
import time


class ActivationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = flow.matmul

    def forward(self, a, b):
        y = self.net(a, b)
        return y


class _TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config.enable_auto_parallel(True)

    def build(self, a, b):
        x = self.model(a, b)
        return x


class TestLazyAllSbpCombinationTesting:
    def test_profile_matmul(self):
        print("Testing...")
        flow.boxing.nccl.enable_use_compute_stream(True)
        for i in range(args.repeat):
            model_diff_hierarchy = ActivationNet()
            graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)
            a = flow.randn((args.n, args.k), device="cuda")
            b = flow.randn((args.k, args.m), device="cuda")
            # x = x.to_global(
            #     sbp=[flow.sbp.split(0)],
            #     placement=flow.placement(type="cuda", ranks=np.array(range(2))),
            # )
            t1 = time.perf_counter()
            with flow.no_grad():
                y = graph_diff_hierarchy(a, b)
            t2 = time.perf_counter()


if __name__ == "__main__":
    m = TestLazyAllSbpCombinationTesting()
    m.test_profile_matmul()
