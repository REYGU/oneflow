import oneflow as flow
import utility


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "ones_like",
        ]
        self.tensor_generator = utility.generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
