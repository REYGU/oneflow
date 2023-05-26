import oneflow as flow
import utility


def generate_mm_input(logmax=30):
    for k in range(logmax):
        for n in range((logmax - k) // 2):
            for m in range(logmax - k - n):
                yield (
                    flow.randn((2**n, 2**k), device="cuda"),
                    flow.randn((2**k, 2**m), device="cuda"),
                )


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "mm",
        ]
        self.tensor_generator = generate_mm_input


if __name__ == "__main__":
    Profiler().run()
