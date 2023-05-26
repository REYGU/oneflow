import oneflow as flow
import utility


def generate_nll_input(logmax=30):
    for i in range(logmax):
        for j in range(0, logmax - i):
            N = 2**i
            C = 2**j
            yield (
                flow.randn((N, C), device="cuda"),
                flow.randint(0, C, (N,), device="cuda"),
            )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            flow.nn.NLLLoss(reduction='none'),
        ]
        self.tensor_generator = generate_nll_input


if __name__ == "__main__":
    Profiler().run()
