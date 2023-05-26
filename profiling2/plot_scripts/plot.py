# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statistics
from scipy import stats
import numpy as np
from scipy import optimize  # 最小二乘法拟合
import matplotlib.pyplot as plt  # python matplotlib 绘图
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图


# from mpl_toolkits.mplot3d import Axes3D


# %%


# %%
df = pd.read_csv(r"results/profile_matmul.20230522024040.csv")
df[['n', 'k', 'm']] = pd.DataFrame(columns=['n', 'k', 'm'], data=df['shape'].map(eval).map(lambda x: (*x[0], x[1][1])).tolist(), index=df.index)
df['nkm'] = df.n*df.k*df.m
df['nk'] = df.n*df.k
df['km'] = df.k*df.m
df

# %%
log2_vars = ["m", "n", "k", "nkm", 'nk', 'km', "time"]
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)
print(df)

# %%


# %%
key = 'nk'
x_key = 'n'
grouped = dict(list(df.groupby(key)))

for idx, (k, sf) in enumerate(grouped.items()):
    # x = sf.log2_nkm - sf[f'log2_{key}']
    x = sf.log2_nkm

    y = sf.log2_time

    plt.figure()
    plt.title(str(k))
    plt.plot(x, y, "o")

    plt.show()

    if idx > 10:
        break


# %%


# %%
def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    p_inter = np.linalg.solve(A, d).T
    (x1, y1, z1) = p_inter[0]
    (x2, y2, z2) = (p_inter + aXb_vec)[0]
    t = ( - p_inter[0][2]) / aXb_vec[2]
    print(p_inter[0], aXb_vec, t)
    return p_inter[0], (p_inter + aXb_vec*t)[0]
    return (y2-y1, x1-x2, -x1*(y2-y1)+y1*(x2-x1))


a, b = (1, -1, 1, 2), (-1, 2, 2, 3)
# a = [param2[0], param2[1], -1, param2[2]]
# b = [param3[0], param3[1], -1, param3[2]]
print(plane_intersect(a, b))

# %%
for row in enumerate(sf.iterrows()):
    print(row) 

# %%


# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statistics
from scipy import stats
import numpy as np
from scipy import optimize  # 最小二乘法拟合
import matplotlib.pyplot as plt  # python matplotlib 绘图
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图


# from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"results/profile_softmax.csv")

df[["m", "n"]] = pd.DataFrame(
    columns=["m", "n"], data=df["shape"].map(eval).tolist(), index=df.index
)
log2_vars = ["m", "n", "time"]
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)
print(df)

grouped = dict(list(df.groupby("op")))
sf = grouped["softmax"]
# sf = grouped["log_softmax"]


def classify_points(tb, lower_bound=4, upper_bound=10):
    labels = [0] * len(tb)
    for index, (_, point) in enumerate(tb.iterrows()):
        if point.time <= lower_bound:
            labels[index] = 1
        elif point.time >= lower_bound:
            neighbor_point = tb[
                (tb.log2_n == point.log2_n) & (tb.log2_m == point.log2_m - 1)
            ]
            if len(neighbor_point) == 0:
                labels[index] = 2
                continue

            # print(point, neighbor_point)
            # print()
            ratio = point.time / neighbor_point.time.values[0]
            labels[index] = 2 if (ratio > 0.8 and ratio < 1.2) else 3
        else:
            pass
    assert len(labels) == len(tb)
    return labels


sf["label"] = classify_points(sf)
sf

# sf[(sf.time > 4) & (sf.ping)]


def plot_3d_points_plt(x, y, z, ax, c):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x, y, z, color=c)


def plot_3d_points(x, y, z, func=None, param=None):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x, y, z, color="r")

    if func:
        x = np.arange(0, 30, 1)  # x定义域，离散
        y = np.arange(0, 30, 1)  # y定义域，离散
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y, param)  # 带入拟合得到的 a, b
        # print(X, Y, Z)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    plt.show()


def plane_func_3var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return a * x + b * y + c


def plane_func_2var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return b * y + c


def plane_func_1var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return c * np.ones_like(x)


def residuals(p, z, x, y, func):
    """得到数据 z 和拟合函数之间的差"""
    return z - func(x, y, p)


def fit_3d_plane(x, y, z, func):
    plsq = optimize.leastsq(
        residuals, np.array([0, 0, 0]), args=(z, x, y, func)
    )  # 最小二乘法拟合
    param = plsq[0]  # 获得拟合结果
    # print("拟合结果:\na = {}".format(a))
    # print("b = {}".format(b))
    # print("c = {}".format(c))
    # plot_3d_points(x, y, z, func, param)
    return param


def fit_3d_plane_sf1(x, y, z):
    def plane_func(x, y, p):
        c = p
        return c * np.ones_like(x)

    plsq = optimize.leastsq(
        residuals, np.array([0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (c,) = plsq[0]  # 获得拟合结果
    param = (0, 0, c)
    return param


def fit_3d_plane_sf2(x, y, z):
    def plane_func(x, y, p):
        b, c = p
        return b * y + c

    plsq = optimize.leastsq(
        residuals, np.array([0, 0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (b, c) = plsq[0]  # 获得拟合结果
    param = (0, b, c)
    return param


def fit_3d_plane_sf3(x, y, z):
    def plane_func(x, y, p):
        (a, b, c) = p
        return a * x + b * y + c

    plsq = optimize.leastsq(
        residuals, np.array([0, 0, 0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (a, b, c) = plsq[0]  # 获得拟合结果
    param = (a, b, c)
    return param


def fit_plane(df):
    params = []
    for plane_id in range(1, 4):
        sub_df = df[df.label == plane_id]
        x = sub_df["log2_m"]
        y = sub_df["log2_n"]
        z = sub_df["log2_time"]
        # param1 = fit_3d_plane_sf1(x, y, z)
        # param = fit_3d_plane(x, y, z, eval(f'fit_3d_plane_sf{plane_id}'))
        param = eval(f"fit_3d_plane_sf{plane_id}")(x, y, z)
        params.append(param)
    return params


(param1, param2, param3) = fit_plane(sf)
# print('param1: ', param1)
# print('param2: ', param2)
# print('param3: ', param3)

# sf1 = sf[sf.label == 1]
# sf2 = sf[sf.label == 2]
# sf3 = sf[sf.label == 3]

# x = sf1["log2_m"]
# y = sf1["log2_n"]
# z = sf1["log2_time"]
# # param1 = fit_3d_plane_sf1(x, y, z)
# param1 = fit_3d_plane(x, y, z, plane_func_1var)


# # x = sf3["log2_m"]
# # y = sf3["log2_n"]
# # z = sf3["log2_time"]
# # fit_3d_plane(x, y, z)

# # c1 = statistics.mode(sf1['log2_time'])
# # print('sf1: ', (0, 0, c1))

# fig = plt.figure(figsize=(8, 6))
# ax = plt.axes(projection="3d")

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")


# plot_3d_points_plt(x, y, z, ax, "r")
# print("sf1: ", param1)

# x = sf2["log2_m"]
# y = sf2["log2_n"]
# z = sf2["log2_time"]
# # param2 = fit_3d_plane_sf2(x, y, z)
# param2 = fit_3d_plane(x, y, z, plane_func_2var)
# plot_3d_points_plt(x, y, z, ax, "g")
# print("sf2: ", param2)

# x = sf3["log2_m"]
# y = sf3["log2_n"]
# z = sf3["log2_time"]
# # param3 = fit_3d_plane_sf3(x, y, z)
# param3 = fit_3d_plane(x, y, z, plane_func_3var)
# plot_3d_points_plt(x, y, z, ax, "b")
# print("sf3: ", param3)


def plane_intersect_projz(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.0]).reshape(3, 1)

    p_inter = np.linalg.solve(A, d).T
    (x1, y1, z1) = p_inter[0]
    (x2, y2, z2) = (p_inter + aXb_vec)[0]

    # return p_inter[0], (p_inter + aXb_vec)[0]
    # return: (A, B, C), z = Ax + By + C
    return (y2 - y1, x1 - x2, -x1 * (y2 - y1) + y1 * (x2 - x1))


# a, b = (1, -1, 1, 2), (-1, 1, 1, 3)
a = [param2[0], param2[1], -1, param2[2]]
b = [param3[0], param3[1], -1, param3[2]]
param4 = plane_intersect_projz(a, b)
print("param1: ", param1)
print("param2: ", param2)
print("param3: ", param3)
print("param4: ", param4)


x = np.arange(0, 30, 1)  # x定义域，离散
y = np.arange(0, 30, 1)  # y定义域，离散
X, Y = np.meshgrid(x, y)
# range1 = (func(X, Y, param2) < 0) & (func(X, Y, param3) < 0)
# range2 = (func(X, Y, param2) < 0) & (func(X, Y, param4) < 0)
# range3 = (func(X, Y, param3) < 0) & (func(X, Y, param4) < 0)


def piecewise_plane_func(x, y, z, param1, param2, param3, param4):
    (a1, b1, c1) = param1
    (a2, b2, c2) = param2
    (a3, b3, c3) = param3
    mask = piecewise_plane_classify(x, y, z, param1, param2, param3, param4)
    value = (
        (a1 * x + b1 * y + c1)
        * mask[0]
        + (a2 * x + b2 * y + c2)
        * mask[1]
        + (a3 * x + b3 * y + c3)
        * mask[2]
    )
    return value


def piecewise_plane_classify(x, y, z, param1, param2, param3, param4):
    mask = [
        ((plane_func_3var(x, y, param2) <= z) & (plane_func_3var(x, y, param3) <= z)),
        ((plane_func_3var(x, y, param2) > z) & (plane_func_3var(x, y, param4) > 0)),
        ((plane_func_3var(x, y, param3) > z) & (plane_func_3var(x, y, param4) < 0)),
    ]
    return mask


# fig = plt.figure(figsize=(8, 6))
# ax = plt.axes(projection="3d")

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")


# print(X, Y, Z)


x = sf["log2_m"]
y = sf["log2_n"]
z = sf["log2_time"]

# pred_z = piecewise_plane_func(x, y, c1, param2, param3, param4)
# print("-" * 150)
# for i, j, k, l in zip(x, y, z, pred_z):
#     print(i, j, k, l)


fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

sf["label2"] = np.argmax(piecewise_plane_classify(sf.log2_m, sf.log2_n, param1[2], param1, param2, param3, param4), 0)
for lid in range(3):
    sub_sf = sf[sf.label2 == lid]
    x = sub_sf["log2_m"]
    y = sub_sf["log2_n"]
    z = sub_sf["log2_time"]
    plot_3d_points_plt(x, y, z, ax, 'rgb'[lid])

x = np.arange(0, 30, 1)  # x定义域，离散
y = np.arange(0, 30, 1)  # y定义域，离散
X, Y = np.meshgrid(x, y)
Z = piecewise_plane_func(X, Y, param1[2], param1, param2, param3, param4)  # 带入拟合得到的 a, b
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")


plt.show()


# %%
np.array(piecewise_plane_classify(sf.log2_m, sf.log2_n, param1[2], param1, param2, param3, param4)).shape

# %%
for i in range(4, 20):
    for j in range(i+1, 20):
        print(f'python .\plot_softmax.py {i} {j}')

# %%
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

df = pd.read_csv(r"results/profile_softmax.csv")

df[["m", "n"]] = pd.DataFrame(
    columns=["m", "n"], data=df["shape"].map(eval).tolist(), index=df.index
)
log2_vars = ["m", "n", "time"]
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)
# print(df)

grouped = dict(list(df.groupby("op")))
# sf = grouped["softmax"]
sf = grouped["log_softmax"]


def classify_points(tb, lower_bound=4, upper_bound=10):
    labels = [0] * len(tb)
    for index, (_, point) in enumerate(tb.iterrows()):
        if point.time <= lower_bound:
            labels[index] = 1
        elif point.time >= upper_bound:
            neighbor_point = tb[
                (tb.log2_n == point.log2_n) & (tb.log2_m == point.log2_m - 1)
            ]
            if len(neighbor_point) == 0:
                labels[index] = 2
                continue

            # print(point, neighbor_point)
            # print()
            ratio = point.time / neighbor_point.time.values[0]
            labels[index] = 2 if (ratio > 0.8 and ratio < 1.2) else 3
        else:
            pass
    assert len(labels) == len(tb)
    return labels


sf["label"] = classify_points(sf)
sf

# sf[(sf.time > 4) & (sf.ping)]


def plot_3d_points_plt(x, y, z, ax, c):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x, y, z, color=c)


def plot_3d_points(x, y, z, func=None, param=None):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x, y, z, color="r")

    if func:
        x = np.arange(0, 30, 1)  # x定义域，离散
        y = np.arange(0, 30, 1)  # y定义域，离散
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y, param)  # 带入拟合得到的 a, b
        # print(X, Y, Z)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    plt.show()


def plane_func_3var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return a * x + b * y + c


def plane_func_2var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return b * y + c


def plane_func_1var(x, y, p):
    """数据拟合所用的函数：z=ax+by
    :param x: 自变量 x
    :param y: 自变量 y
    :param p: 拟合参数 a, b
    """
    a, b, c = p
    return c * np.ones_like(x)


def residuals(p, z, x, y, func):
    """得到数据 z 和拟合函数之间的差"""
    return z - func(x, y, p)


def fit_3d_plane(x, y, z, func):
    plsq = optimize.leastsq(
        residuals, np.array([0, 0, 0]), args=(z, x, y, func)
    )  # 最小二乘法拟合
    param = plsq[0]  # 获得拟合结果
    # print("拟合结果:\na = {}".format(a))
    # print("b = {}".format(b))
    # print("c = {}".format(c))
    # plot_3d_points(x, y, z, func, param)
    return param


def fit_3d_plane_sf1(x, y, z):
    def plane_func(x, y, p):
        c = p
        return c * np.ones_like(x)

    plsq = optimize.leastsq(
        residuals, np.array([0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (c,) = plsq[0]  # 获得拟合结果
    param = (0, 0, c)
    return param


def fit_3d_plane_sf2(x, y, z):
    # def plane_func(x, y, p):
    #     b, c = p
    #     return b * y + c

    # plsq = optimize.leastsq(
    #     residuals, np.array([0, 0]), args=(z, x, y, plane_func)
    # )  # 最小二乘法拟合
    # (b, c) = plsq[0]  # 获得拟合结果
    # param = (0, b, c)
    # return param
    def plane_func(x, y, p):
        (a, b, c) = p
        return a * x + b * y + c

    plsq = optimize.leastsq(
        residuals, np.array([0, 0, 0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (a, b, c) = plsq[0]  # 获得拟合结果
    param = (a, b, c)
    return param


def fit_3d_plane_sf3(x, y, z):
    def plane_func(x, y, p):
        (a, b, c) = p
        return a * x + b * y + c

    plsq = optimize.leastsq(
        residuals, np.array([0, 0, 0]), args=(z, x, y, plane_func)
    )  # 最小二乘法拟合
    (a, b, c) = plsq[0]  # 获得拟合结果
    param = (a, b, c)
    return param


def fit_piecewise_plane(df):
    params = []
    for plane_id in range(1, 4):
        sub_df = df[df.label == plane_id]
        x = sub_df["log2_m"]
        y = sub_df["log2_n"]
        z = sub_df["log2_time"]
        # param1 = fit_3d_plane_sf1(x, y, z)
        # param = fit_3d_plane(x, y, z, eval(f'fit_3d_plane_sf{plane_id}'))
        param = eval(f"fit_3d_plane_sf{plane_id}")(x, y, z)
        params.append(param)
    return params


def plane_intersect_projz(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.0]).reshape(3, 1)

    p_inter = np.linalg.solve(A, d).T
    (x1, y1, z1) = p_inter[0]
    (x2, y2, z2) = (p_inter + aXb_vec)[0]

    # return p_inter[0], (p_inter + aXb_vec)[0]
    # return: (A, B, C), z = Ax + By + C
    return (y2 - y1, x1 - x2, -x1 * (y2 - y1) + y1 * (x2 - x1))

def piecewise_plane_func(x, y, param1, param2, param3, param4):
    (a1, b1, c1) = param1
    (a2, b2, c2) = param2
    (a3, b3, c3) = param3
    mask = piecewise_plane_classify(x, y, param1, param2, param3, param4)
    value = (
        (a1 * x + b1 * y + c1)
        * mask[0]
        + (a2 * x + b2 * y + c2)
        * mask[1]
        + (a3 * x + b3 * y + c3)
        * mask[2]
    )
    return value


def piecewise_plane_classify(x, y, param1, param2, param3, param4):
    z = param1[2]
    mask = [
        ((plane_func_3var(x, y, param2) <= z) & (plane_func_3var(x, y, param3) <= z)),
        ((plane_func_3var(x, y, param2) > z) & (plane_func_3var(x, y, param4) > 0)),
        ((plane_func_3var(x, y, param3) > z) & (plane_func_3var(x, y, param4) < 0)),
    ]
    return mask


def eval_picecwise_plane(df, params):
    
    x = df["log2_m"]
    y = df["log2_n"]
    z = df["log2_time"]

    pred_z = piecewise_plane_func(x, y, *params)
    # print("-" * 150)
    # for i, j, k, l in zip(x, y, z, pred_z):
    #     print(i, j, k, l)

    print("-" * 150)
    # print(LOWER_BOUND, UPPER_BOUND)
    print('average: ', sum(np.abs(pred_z - z))/len(z))
    print("-" * 150)


(param1, param2, param3) = fit_piecewise_plane(sf)
# a, b = (1, -1, 1, 2), (-1, 1, 1, 3)
a = [param2[0], param2[1], -1, param2[2]]
b = [param3[0], param3[1], -1, param3[2]]
param4 = plane_intersect_projz(a, b)
params = [param1, param2, param3, param4]

# print("param1: ", param1)
# print("param2: ", param2)
# print("param3: ", param3)
# print("param4: ", param4)

eval_picecwise_plane(sf, params)

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

sf["label2"] = np.argmax(piecewise_plane_classify(sf.log2_m, sf.log2_n, *params), 0)
for lid in range(3):
# for lid in range(3):
    sub_sf = sf[sf.label2 == lid]
    x = sub_sf["log2_m"]
    y = sub_sf["log2_n"]
    z = sub_sf["log2_time"]
    plot_3d_points_plt(x, y, z, ax, 'rgb'[lid])

x = np.arange(0, 30, 1)  # x定义域，离散
y = np.arange(0, 30, 1)  # y定义域，离散
X, Y = np.meshgrid(x, y)
Z = piecewise_plane_func(X, Y, *params)  # 带入拟合得到的 a, b
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")


plt.show()


# %%
sf

# %%
x, y = 131072, 1024


def predict(x, y):
    log2x = np.log2(x)
    log2y = np.log2(y)
    log2z = piecewise_plane_func(log2x, log2y, *params)
    tarz = sf[(sf.m == x) & (sf.n == y)]
    return np.power(2, log2z), log2z,tarz.time.tolist(), tarz.log2_time.tolist()


predict(x, y)

# %%
predict(524288, 1024)

# %%
sf[(sf.m == 524288 )&(sf.n == 1024) ]

# %%
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

df = pd.read_csv(r"results/profile_math_unary.csv")

# %%
df['m'] = df['shape'].map(eval).map(lambda x:x[0])

log2_vars = ["m","time"]
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)

# %%
df

# %%
grouped = dict(list(df.groupby('op')))

# %%
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


curve_params = []
for op in grouped:
    x = np.array(grouped[op]["log2_m"])
    y = np.array(grouped[op]["log2_time"])
    p, e = optimize.curve_fit(piecewise_linear, x, y, bounds=(0, [20, 20, 2, 2]))
    xd = np.linspace(0, 30, 100)
    curve_params.append((op, p, "piecewise", grouped[op]))
    plt.figure()
    plt.title(op)
    plt.plot(x, y, "o")
    plt.plot(xd, piecewise_linear(xd, *p))
    # break


# su = pd.DataFrame(columns=['op', 'param', 'type', 'raw'], data=curve_params).set_index('op')
# su.to_pickle('curve.pkl')

# %%
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

sub_df = df[(df.op != 'digamma') & (df.op != 'lgamma') ]
# sub_df = df
# sub_df = df[(df.op != 'digamma') & (df.op != 'lgamma') & (df.op != 'asinh') & (df.op != 'acosh')]
# sub_df = sub_df[(sub_df.time <= 4) | (sub_df.time >= 20)]
x = np.array(sub_df["log2_m"])
y = np.array(sub_df["log2_time"])
p, e = optimize.curve_fit(piecewise_linear, x, y, bounds=(0, [20, 20, 2, 2]))
xd = np.linspace(0, 30, 100)
curve_params.append((op, p, "piecewise", grouped[op]))
plt.figure()
plt.title(op)
plt.plot(x, y, "o")
plt.plot(xd, piecewise_linear(xd, *p))
    # break

print(list(p))
# su = pd.DataFrame(columns=['op', 'param', 'type', 'raw'], data=curve_params).set_index('op')
# su.to_pickle('curve.pkl')

# %%
# res = []
# for op in grouped:
#     x = np.array(grouped[op]["log2_m"])
#     y = np.array(grouped[op]["log2_time"])
#     pred = piecewise_linear(x, *p)
#     loss = sum(np.abs(pred-y))/len(y)
#     res.append((op, loss))
#     print(op, loss)
    

# %%
sorted(res, key=lambda x:x[1])

# %%
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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"results/profile_matmul.20230524023628.csv")
df = df[df.time > 4]
df[['i', 'k', 'j']] = pd.DataFrame(columns=['i', 'k', 'j'], data=df['shape'].map(eval).map(lambda x: (*x[0], x[1][1])).tolist(), index=df.index)
df[['X']] = pd.DataFrame(columns=['X'], data=df['shape'].map(eval).map(lambda x: np.array((*x[0], x[1][1]))).to_numpy(), index=df.index)

df['ij'] = df.i * df.j
df['jk'] = df.k * df.j
df['ik'] = df.i * df.k
df['ijk'] = df.i * df.j * df.k

log2_vars = ['i', 'k','j', 'ij', 'jk', 'ik', 'ijk', "time", 'X']
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)
print(df.head(10))


# X = np.random.rand(100,2)




# %%
x_key = 'log2_X'
z_key = 'log2_time'
z = df[z_key]
x = np.array(df[x_key].tolist())
# print(x[:10])


poly = PolynomialFeatures(degree=1)
print(poly)
X_t = poly.fit_transform(x)
# print(x[:5])
# print(X_t[:5])

clf = LinearRegression()

clf.fit(X_t, z)
print(len(clf.coef_), clf.coef_)
print(clf.intercept_)

# %%
len(df)

# %%
sum((abs(sp.pred_time - sp.time)))/len(sp)

# %%


# %%
len(df)

# %%
sp = df
pred_x = np.array(sp[x_key].tolist())
pred_tx = poly.fit_transform(pred_x)
pred_time = clf.predict(pred_tx)
pred_time
sp['log2_pred_time'] = pred_time
sp['pred_time'] = 2 ** (pred_time)
# sp['pred_time'] = pred_time
# sp['log2_pred_time'] = np.log2(pred_time)

sp[['shape', 'i', 'j', 'k', 'log2_time', 'log2_pred_time', 'time', 'pred_time']]

print(sum((abs(sp.pred_time - sp.time )))/len(sp))
print(sum((abs(sp.log2_pred_time - sp.log2_time )))/len(sp))
# print(sum((abs(sp.pred_time - sp.time)))/len(sp))
# print(sum(np.sqrt((sp.pred_time - sp.time)*(sp.pred_time - sp.time)))/len(sp))

# %%
sp = df.sample(100)
pred_x = np.array(sp[x_key].tolist())
pred_tx = poly.fit_transform(pred_x)
pred_time = clf.predict(pred_tx)
pred_time
# sp['pred_time'] = pred_time
# sp['log2_pred_time'] = np.log2(pred_time)
sp['log2_pred_time'] = pred_time
sp['pred_time'] = 2 ** (pred_time)

sp[['shape', 'i', 'j', 'k', 'log2_time', 'log2_pred_time', 'time', 'pred_time']]



# %%


# %%
grouped = 

x = np.arange(0, 30, 1)  # x定义域，离散
y = np.arange(0, 30, 1)  # y定义域，离散
X, Y = np.meshgrid(x, y)
Z = piecewise_plane_func(X, Y, *params)  # 带入拟合得到的 a, b
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")


plt.show()



