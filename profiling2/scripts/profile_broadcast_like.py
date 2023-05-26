import oneflow as flow
import utility


def generate_broadcast_like_input(logmax=30):
    for i in range(logmax):
        for j in range(1, logmax-i):
            yield (
                flow.randn((2**i, 1), device="cuda"),
                flow.randn((2**i, 2**j), device="cuda"),
            )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "broadcast_like",
        ]
        self.tensor_generator = generate_broadcast_like_input


if __name__ == "__main__":
    Profiler().run()
