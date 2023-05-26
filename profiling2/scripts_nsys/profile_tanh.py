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
parser.add_argument("--shape", nargs="+", type=int, default=8)
# parser.add_argument('--dtype', type=int, default=100)
args = parser.parse_args()


import oneflow as flow
from oneflow import nn
import os
import numpy as np
import time
import oneflow.unittest


class _TestModuleDiffHierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Tanh()

    def forward(self, x):
        y = self.net(x)
        return y


class _TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config.enable_auto_parallel(True)

    def build(self, x):
        x = self.model(x)
        return x


class TestLazyAllSbpCombinationTesting:
    # def test_profile_tanh(self):
    #     flow.boxing.nccl.enable_use_compute_stream(True)
    #     for i in range(args.repeat):
    #         x = flow.randn(args.shape, device="cuda:0")
    #         x = x.to_global(
    #             sbp=[flow.sbp.split(0)],
    #             placement=flow.placement(type="cuda", ranks=np.array(range(2))),
    #         )

    #         model_diff_hierarchy = _TestModuleDiffHierarchy()
    #         graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)

    #         # print("Start to run graph_diff_hierarchy")
    #         # record the start time in python
    #         t1 = time.perf_counter()
    #         with flow.no_grad():
    #             y = graph_diff_hierarchy(x)
    #         t2 = time.perf_counter()
    #         # print(y.to_local().numpy())
    #         # print("time used: ", t2 - t1)
    def test_profile_tanh(self):
        flow.boxing.nccl.enable_use_compute_stream(True)
        for i in range(args.repeat):
            model_diff_hierarchy = _TestModuleDiffHierarchy()
            graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)

            print("repeat...: ", i)
            x = flow.randn(args.shape, device="cuda:0")
            x = x.to_global(
                sbp=[flow.sbp.split(0)],
                placement=flow.placement(type="cuda", ranks=np.array(range(2))),
            )
            t1 = time.perf_counter()
            with flow.no_grad():
                y = graph_diff_hierarchy(x)
            t2 = time.perf_counter()

    def test_profile_tanh_1n1d(self):
        flow.boxing.nccl.enable_use_compute_stream(True)
        model_diff_hierarchy = _TestModuleDiffHierarchy()
        graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)

        for i in range(args.repeat):
            # print("repeat...: ", i)
            x = flow.randn(args.shape, device="cuda:0")
            t1 = time.perf_counter()
            with flow.no_grad():
                y = graph_diff_hierarchy(x)
            t2 = time.perf_counter()


if __name__ == "__main__":
    # unittest.main()
    TestLazyAllSbpCombinationTesting().test_profile_tanh_1n1d()
    # TestLazyAllSbpCombinationTesting().test_profile_tanh()
