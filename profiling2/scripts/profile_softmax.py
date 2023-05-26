import oneflow as flow
import utility


def generate_softmax_input(logmax=30):
    for i in range(logmax):
        for j in range(logmax - i):
            yield flow.randn((2**j, 2**i), device="cuda")


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "softmax",
            "log_softmax",
        ]
        self.tensor_generator = generate_softmax_input


if __name__ == "__main__":
    Profiler().run()
