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
args = parser.parse_args()


import oneflow as flow
from oneflow import nn
import os
import numpy as np
import time


class ActivationNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

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
    def test_profile_activation(self, net):
        print("Testing...", net)
        flow.boxing.nccl.enable_use_compute_stream(True)
        for i in range(args.repeat):
            x = flow.randn(args.shape, device="cuda:0")
            x = x.to_global(
                sbp=[flow.sbp.split(0)],
                placement=flow.placement(type="cuda", ranks=np.array(range(2))),
            )

            model_diff_hierarchy = ActivationNet(net)
            graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)

            t1 = time.perf_counter()
            with flow.no_grad():
                y = graph_diff_hierarchy(x)
            t2 = time.perf_counter()

    def test_profile_relu(self):
        self.test_profile_activation(nn.ReLU())

    def test_profile_hardshrink(self):
        self.test_profile_activation(nn.Hardshrink())

    def test_profile_silu(self):
        self.test_profile_activation(nn.SiLU())

    def test_profile_selu(self):
        self.test_profile_activation(nn.SELU())

    def test_profile_softsign(self):
        self.test_profile_activation(nn.Softsign())

    def test_profile_hardsigmoid(self):
        self.test_profile_activation(nn.Hardsigmoid())

    def test_profile_hardswish(self):
        self.test_profile_activation(nn.Hardswish())

    def test_profile_softplus(self):
        self.test_profile_activation(nn.Softplus())

    def test_profile_elu(self):
        self.test_profile_activation(nn.ELU())

    def test_profile_celu(self):
        self.test_profile_activation(nn.CELU())

    def test_profile_mish(self):
        self.test_profile_activation(nn.Mish())

    def test_profile_gelu(self):
        self.test_profile_activation(nn.GELU())

    def test_profile_tanh(self):
        self.test_profile_activation(nn.Tanh())


if __name__ == "__main__":
    m = TestLazyAllSbpCombinationTesting()
    m.test_profile_relu()
