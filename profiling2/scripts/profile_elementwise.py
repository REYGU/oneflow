import oneflow as flow
import utility


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            # flow.nn.Dropout(),
            # flow._C.identity,
            # flow.cast
            lambda x: flow.cast(x, flow.float64)
        ]
        self.tensor_generator = utility.generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
