import oneflow as flow
import utility


def generate_one_tensor(logmax=25):
    for i in range(logmax):
        yield flow.randn(2**i, device="cuda")

    for n in [82 * 1536, 82 * 1536 * 2, 82 * 1536 * 4, 82 * 1536 * 8]:
        yield flow.randn(n, device="cuda")


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            flow.nn.Sigmoid(),
        ]
        self.tensor_generator = generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
