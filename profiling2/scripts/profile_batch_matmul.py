import oneflow as flow
import utility


def generate_mm_input(logmax=30):
    for b in range(1, 20):
        for k in range(logmax - b):
            for n in range((logmax - k - b) // 2):
                for m in range(logmax - k - n - b):
                    yield (
                        flow.randn((2**b, 2**n, 2**k), device="cuda"),
                        flow.randn((2**b, 2**k, 2**m), device="cuda"),
                    )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "bmm",
        ]
        self.tensor_generator = generate_mm_input


if __name__ == "__main__":
    Profiler().run()
