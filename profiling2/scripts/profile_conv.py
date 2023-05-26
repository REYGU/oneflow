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
from oneflow.test_utils.automated_test_util import *
import oneflow as flow


def _generate_two_broadcast_input(logmax=30):
    for i in range(logmax):
        yield (torch.randn(2**i), torch.randn(2**i))

    for i in range(logmax):
        for j in range(logmax - i):
            yield (torch.randn((2**j, 2**i)), torch.randn((2**i)))

    for i in range(logmax):
        for j in range(logmax - i):
            yield (torch.randn((2**i)), torch.randn((2**j, 2**i)))


def _generate_two_input(logmax=30):
    for i in range(logmax):
        yield (torch.randn(2**i), torch.randn(2**i))


@flow.unittest.skip_unless_1n1d()
class ProfilingModule(flow.unittest.TestCase):
    @profile(torch.add)
    def profile_add(test_case):
        for a, b in _generate_two_broadcast_input():
            torch.add(a, b)


if __name__ == "__main__":
    unittest.main()
