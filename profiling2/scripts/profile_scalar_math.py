import oneflow as flow
import utility



class Profiler(utility.ProfilerBase):
    def __init__(self):
        self.op_list = [
            lambda x: x + 5.0, 
            lambda x: x - 5.0, 
            lambda x: x * 5.0, 
            lambda x: x / 5.0, 
            lambda x: x % 5.0, 
            lambda x: x ** 5.0, 
            lambda x: 5.0 ** x, 
            lambda x: x > 5,
            lambda x: x < 5,
            lambda x: x == 5,
            lambda x: x >= 5,
            lambda x: x <= 5,
            lambda x: x != 5,
            lambda x: x | 5,
            lambda x: x & 5,
            lambda x: x ^ 5,
        ]
        self.tensor_generator = utility.generate_one_tensor


if __name__ == "__main__":
    Profiler().run()
