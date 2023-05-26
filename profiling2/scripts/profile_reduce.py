import oneflow as flow
import utility

def generate_reduce_input(logmax=30):
    for i in range(0, logmax):
        for j in range(1, logmax - i):
            yield flow.randn((2**i, 2**j), device="cuda")


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            # lambda x: flow._C.reduce_mean(x, [-1]),
            lambda x: flow._C.reduce_all(x, [-1]),
            lambda x: flow._C.reduce_any(x, [-1]),
            lambda x: flow._C.reduce_prod(x, [-1]),
            lambda x: flow._C.reduce_max(x, [-1]),
            lambda x: flow._C.reduce_min(x, [-1]),
            lambda x: flow._C.reduce_nansum(x, [-1]),
            lambda x: flow._C.reduce_sum(x, [-1]),
        ]
        self.tensor_generator = generate_reduce_input


if __name__ == "__main__":
    Profiler().run()
