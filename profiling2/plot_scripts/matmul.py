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


# X = np.random.rand(100,2)

# %%
df = pd.read_csv(r"results/profile_matmul.20230524101643.csv")
# df = pd.read_csv(r"results/profile_matmul.20230524023628.csv")
df = df[df.time > 5]
df[["i", "k", "j"]] = pd.DataFrame(
    columns=["i", "k", "j"],
    data=df["shape"].map(eval).map(lambda x: (*x[0], x[1][1])).tolist(),
    index=df.index,
)
df[["X"]] = pd.DataFrame(
    columns=["X"],
    data=df["shape"].map(eval).map(lambda x: np.array((*x[0], x[1][1]))).to_numpy(),
    index=df.index,
)

# df['ij'] = df.i * df.j
# df['jk'] = df.k * df.j
# df['ik'] = df.i * df.k
# df['ijk'] = df.i * df.j * df.k

log2_vars = ["time", "X"]
# log2_vars = ['i', 'k','j', 'ij', 'jk', 'ik', 'ijk', "time", 'X']
for v in log2_vars:
    df[f"log2_{v}"] = df[v].map(np.log2)
print(df.sample(10))

# %%
# df['count2'] = df['count'].apply(lambda x: max(eval(x)))
# df['time2'] = df.time * df.valid / df.count2
# df[['shape', 'i', 'j', 'k', 'time', 'pred_time', 'valid', 'count2', 'time2']].to_csv('test.csv', index=False)

# %%
x_key = 'log2_X'
z_key = 'log2_time'
z = df[z_key]
x = np.array(df[x_key].tolist())
# print(x[:10])


poly = PolynomialFeatures(degree=3)
print(poly)
X_t = poly.fit_transform(x)
# print(x[:5])
# print(X_t[:5])

clf = LinearRegression()

clf.fit(X_t, z)
print(len(clf.coef_), clf.coef_)
print(clf.intercept_)



# %%
pred_x = np.array(df[x_key].tolist())
pred_time = clf.predict(poly.fit_transform(pred_x))
pred_time

if x_key == 'log2_X':
    df['log2_pred_time'] = pred_time
    df['pred_time'] = 2 ** (pred_time)
else:
    df['pred_time'] = pred_time
    df['log2_pred_time'] = np.log2(pred_time)



print('raw abs:', sum((abs(df.pred_time - df.time )))/len(df))
print('log abs:', sum((abs(df.log2_pred_time - df.log2_time )))/len(df))
print('raw rel:', sum((abs((df.pred_time - df.time)/df.time )))/len(df))
print('log rel:', sum((abs((df.log2_pred_time - df.log2_time)/df.log2_time )))/len(df))

df.sample(10)[['shape', 'i', 'j', 'k', 'log2_time', 'log2_pred_time', 'time', 'pred_time']]

# %%
drawI = 32 * 32
x = np.arange(0, 25, 1)  # x定义域，离散
y = np.arange(0, 25, 1)  # y定义域，离散
J, K = np.meshgrid(x, y)
I = np.full_like(J, np.log2(drawI))
drawX = (np.concatenate((I.reshape(-1, 1), J.reshape(-1, 1), K.reshape(-1, 1)), 1))
drawZ = clf.predict(poly.fit_transform(drawX))

# drawX = 
# X, Y
# clf.predict(poly.fit_transform(pred_x))
# # Z = piecewise_plane_func(X, Y, *params)  # 带入拟合得到的 a, b

Z = drawZ.reshape(J.shape)




fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")

sub_df = df[df.i == drawI]
x = sub_df.j.apply(np.log2)
y = sub_df.k.apply(np.log2)
z = sub_df.time.apply(np.log2)
ax.scatter(x, y, z, color='r')

ax.set_xlabel("j")
ax.set_ylabel("k")
ax.set_zlabel("z")
ax.plot_surface(J, K, Z, rstride=1, cstride=1, cmap="rainbow")
plt.show()


# %%


# %%



