import oneflow as flow
import utility


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            flow.nn.CELU(),
            flow.nn.ELU(),
            flow.nn.GELU(),
            flow.nn.Hardshrink(),
            flow.nn.Hardsigmoid(),
            flow.nn.Hardswish(),
            flow.nn.Hardtanh(),
            flow.nn.LeakyReLU(),
            flow.nn.LogSigmoid(),
            flow.nn.Mish(),
            flow.nn.ReLU(),
            flow.nn.ReLU6(),
            flow.nn.SELU(),
            flow.nn.SiLU(),
            flow.nn.Sigmoid(),
            flow.nn.Softplus(),
            flow.nn.Softshrink(),
            flow.nn.Softsign(),
            flow.nn.Tanh(),
            flow.nn.Threshold(threshold=0.1, value=20),
        ]
        self.tensor_generator = utility.generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
