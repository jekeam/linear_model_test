import numpy as np
# from sklearn.svm import SVR
from sklearn import linear_model
import pandas as pd


def reject_outliers(data, m=2):
    data = np.asarray(data)
    mean = np.mean(data)
    if np.std(data) == 0:
        std = mean
    else:
        std = np.std(data)
    return data[abs(data - mean) < m * std].tolist()


# del zerro values in line

# del zerro values in line
def del_zerro_in_line(val_arr: list, line_arr: list):
    l = val_arr.copy()

    if len(val_arr) > len(line_arr):
        val_arr = val_arr[-len(line_arr):]
    elif len(line_arr) > len(val_arr):
        line_arr = line_arr[-len(val_arr):]

    slip = 0
    for idx, val in enumerate(l):
        if val == 0:
            val_arr.pop(idx - slip)
            line_arr.pop(idx - slip)
            slip += 1

    return val_arr, line_arr


def check_kof_on_outliers(kof: float, arr: list):
    if kof in reject_outliers(arr):
        return kof
    else:
        return 0


def predict_vect(x: list, y: list, show_chart: bool = False) -> str:
    # y, x = del_zerro_in_lean(y, x)

    kof_cur = y[-1]

    arr = list(zip(x, y))
    x = []
    y = []
    n = 1
    for p in arr:
        for t in range(0, p[0]):
            x.append(n)
            y.append(p[1])
            n += 1

    print('Если = 0, то бырос: ' + str(check_kof_on_outliers(kof_cur, y)))
    # regr = SVR(gamma='scale', C=1.0, epsilon=0.001)
    # regr = SVR( C=1.0, epsilon=0.001)
    regr = linear_model.LinearRegression()
    xr = np.asarray(x).reshape(len(x), 1)
    yr = np.asarray(y).reshape(len(y), 1)
    regr.fit(xr, yr)

    x_predict = n + 1
    kof_predict1 = regr.predict([[n]])[0]
    kof_predict2 = regr.predict([[x_predict]])[0]

    if show_chart:
        import matplotlib.pyplot as plt

        y_predict = kof_predict2
        z = 40

        plt.scatter(xr, yr, color='cyan', marker=',')

        plt.plot(xr, regr.predict(xr), color='black', linestyle='dashed', markersize=2)
        plt.scatter(x_predict, y_predict, s=z * 1, alpha=0.5)
        plt.show()

    print('{}->{}. {}'.format(kof_predict1, kof_predict2, kof_cur))
    if kof_predict2 > kof_predict1:
        return 'UP'
    else:
        return 'DOWN'


if __name__ == '__main__':
    x = [83, 89, 38, 53, 61, 173, 37, 171, 59, 59, 6, 5, 81, 83, 29, 29, 179, 59, 27, 90, 59, 29, 88, 29, 59, 88, 21, 45, 51, 49, 39, 88, 29, 89, 29, 29, 88, 29, 89, 28, 19, 75, 21, 9, 5, 117, 109, 609, 166, 165, 119, 59, 11, 68, 1, 1]
    y = [1.27, 1.28, 1.3, 1.32, 1.35, 1.4, 1.42, 1.45, 1.47, 1.48, 1.5, 1.6, 0, 1.6, 1.62, 1.63, 1.65, 1.67, 1.68, 1.7, 1.72, 1.73, 1.75, 1.77, 1.78, 1.8, 1.82, 0, 1.85, 1.87, 1.88, 1.9, 1.93, 1.95, 1.97, 1.98, 2.0, 2.02, 2.05, 2.07, 2.08, 0, 2.13,
         2.15, 2.2, 2.25, 0, 2.25, 2.4, 2.5, 2.55, 2.6, 2.65, 0, 2.7, 2.65]
    x2 = [1]
    y2 = [1]

    print(predict_vect(x, y, True))
    print(predict_vect(x2, y2, True))
