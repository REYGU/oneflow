import oneflow as flow
import utility


def generate_input(logmax=30):
    for i in range(logmax):
        for j in range(logmax-i):
            yield (flow.randn((2**i, 2**j), device="cuda"), 
                   flow.randn((2**j, ), device="cuda"))


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            lambda x, y: flow._C.fused_bias_add_gelu(x, y, axis=1),
        ]
        self.tensor_generator = generate_input


if __name__ == "__main__":
    Profiler().run()
