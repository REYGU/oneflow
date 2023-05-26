import utility


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "atan2",
        ]
        self.tensor_generator = utility.generate_two_tensor


if __name__ == "__main__":
    Profiler().run()
