import oneflow as flow
import utility


def generate_input(logmax=30):
    for i in range(logmax):
        for j in range(0, logmax-i):
            yield (
                flow.randn((2**i, 2**j), device="cuda"),
            )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            lambda x: flow.nn.functional.layer_norm(x, x.shape[1:]),
        ]
        self.tensor_generator = generate_input


if __name__ == "__main__":
    Profiler().run()
