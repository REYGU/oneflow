import oneflow as flow
import utility


def _generate_two_broadcast_input(logmax=30):
    for i in range(logmax):
        yield (flow.randn(2**i, device="cuda"), flow.randn(2**i, device="cuda"))

    for i in range(logmax):
        for j in range(1, logmax - i):
            yield (
                flow.randn((2**j, 2**i), device="cuda"),
                flow.randn((1, 2**i), device="cuda"),
            )

    for i in range(logmax):
        for j in range(1, logmax - i):
            yield (
                flow.randn((1, 2**i), device="cuda"),
                flow.randn((2**j, 2**i), device="cuda"),
            )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            flow._C.less,
            flow._C.equal,
            flow._C.greater_equal,
            flow._C.less_equal,
            flow._C.not_equal,
            flow._C.logical_or,
            flow._C.logical_and,
            flow._C.logical_xor,
        ]
        self.tensor_generator = _generate_two_broadcast_input


if __name__ == "__main__":
    Profiler().run()
