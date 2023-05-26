import oneflow as flow
import utility


class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "ceil",
            "cos",
            "cosh",
            "digamma",
            # "trigamma",
            "erf",
            "erfc",
            "exp",
            "exp2",
            "expm1",
            "floor",
            "lgamma",
            "log",
            "log2",
            "log10",
            "log1p",
            "nn.functional.logsigmoid",
            "negative",
            "reciprocal",
            # "reciprocal_no_nan",
            # "rint",
            "round",
            "rsqrt",
            # "sigmoid",
            "sign",
            "sin",
            "sinh",
            "sqrt",
            "square",
            "tan",
            # "not_equal_zero",
        ]
        self.tensor_generator = utility.generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
