import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statistics
from scipy import stats
import numpy as np
from scipy import optimize  # 最小二乘法拟合
import matplotlib.pyplot as plt  # python matplotlib 绘图
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图
import sys

# LOWER_BOUND = int(sys.argv[1])
# UPPER_BOUND = int(sys.argv[2])
# from mpl_toolkits.mplot3d import Axes3D
# print(LOWER_BOUND, UPPER_BOUND)

df = pd.read_csv(r"results/profile_matmul.20230524101643.csv")
df[["i", "k", "j"]] = pd.DataFrame(
    columns=["i", "k", "j"],
    data=df["shape"].map(eval).map(lambda x: (*x[0], x[1][1])).tolist(),
    index=df.index,
)

df['m'] = df.i*df.k  + df.j*df.k + df.i*df.j

log2_vars = ["m","time"]
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)


grouped = dict(list(df.groupby('op')))


from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


def piecewise_linear(x, x0, y0, k1, k2):
    # x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
    # x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
    return np.piecewise(
        x,
        [x < x0, x >= x0],
        [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0],
    )

plt.figure()

for op in grouped:
    sub_df = grouped[op]
    x = np.array(sub_df["log2_m"])
    y = np.array(sub_df["log2_time"])
    p, e = optimize.curve_fit(piecewise_linear, x, y, bounds=(0, [20, 20, 2, 2]))
    xd = np.linspace(0, 30, 100)
    plt.title(op)
    plt.plot(x, y, "o")
    plt.plot(xd, piecewise_linear(xd, *p))
    print(op,'\t', list(p))
    
    pred_time = piecewise_linear(x, *p)
    df['log2_pred_time'] = pred_time
    df['pred_time'] = 2 ** (pred_time)
    print('raw abs:', sum((abs(df.pred_time - df.time )))/len(df))
    print('log abs:', sum((abs(df.log2_pred_time - df.log2_time )))/len(df))
    print('raw rel:', sum((abs((df.pred_time - df.time)/df.time )))/len(df))
    print('log rel:', sum((abs((df.log2_pred_time - df.log2_time)/df.log2_time )))/len(df))
    print(df.sample(10)[['shape', 'i', 'j', 'k', 'log2_time', 'log2_pred_time', 'time', 'pred_time']])

plt.show()
    # break


# su = pd.DataFrame(columns=['op', 'param', 'type', 'raw'], data=curve_params).set_index('op')
# su.to_pickle('curve.pkl')


# from scipy import optimize
# import matplotlib.pyplot as plt
# import numpy as np


# def piecewise_linear(x, x0, y0, k1, k2):
#     # x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
#     # x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
#     return np.piecewise(
#         x,
#         [x < x0, x >= x0],
#         [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0],
#     )

# sub_df = df[(df.op != 'digamma') & (df.op != 'lgamma') ]
# # sub_df = df
# # sub_df = df[(df.op != 'digamma') & (df.op != 'lgamma') & (df.op != 'asinh') & (df.op != 'acosh')]
# # sub_df = sub_df[(sub_df.time <= 4) | (sub_df.time >= 20)]
# x = np.array(sub_df["log2_m"])
# y = np.array(sub_df["log2_time"])
# p, e = optimize.curve_fit(piecewise_linear, x, y, bounds=(0, [20, 20, 2, 2]))
# xd = np.linspace(0, 30, 100)
# curve_params.append((op, p, "piecewise", grouped[op]))
# plt.figure()
# plt.title(op)
# plt.plot(x, y, "o")
# plt.plot(xd, piecewise_linear(xd, *p))
#     # break

# print(list(p))
# # su = pd.DataFrame(columns=['op', 'param', 'type', 'raw'], data=curve_params).set_index('op')
# # su.to_pickle('curve.pkl')