import numpy as np
# from sklearn.svm import SVR
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd


def reject_outliers(data, m=2):
    data = np.asarray(data)
    # mean = np.mean(data)
    # if np.std(data) == 0:
    #     std = mean
    # else:
    #     std = np.std(data)
    return data[abs(data - np.mean(data)) <= m * np.std(data)].tolist()


def get_std(data):
    data = np.asarray(data)
    return np.std(data).tolist()


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


def get_vect(x, y, x2, y2):
    # print(list(reversed(y)).index(0))
    # print(list(reversed(y2)).index(0))

    y = y[-len(x):]
    y2 = y2[-len(x2):]

    kof_cur1 = y[-1]
    kof_cur2 = y2[-1]

    arr = list(zip(x, y))
    x = []
    y = []
    n = 1
    for p in arr:
        for t in range(1, p[0]):
            x.append(n)
            y.append(p[1])
            n += 1

    arr2 = list(zip(x2, y2))
    x2 = []
    y2 = []
    n2 = 1
    for p2 in arr2:
        for t2 in range(1, p2[0]):
            x2.append(n2)
            y2.append(p2[1])
            n2 += 1

    regr = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()
    # regr = SVR(gamma='scale', C=1.0, epsilon=0.001)
    # regr2 = SVR(gamma='scale', C=1.0, epsilon=0.001)

    if x > x2:
        x2 = x[-len(x2):]
        n3 = min(x2)
    else:
        x = x2[-len(x):]
        n3 = min(x)

    x_max = max(x)
    x2_max = max(x2)

    p = list(zip(reversed(y), reversed(y2)))

    live = 40
    for k in reversed(p):
        k1, k2 = k[0], k[1]
        if k1 and k2:
            l = 1 / k1 + 1 / k2
            if l < 1:
                proc = (1 - l) * 100
                if proc > 0:
                    if proc >= 3:
                        color = 'pink'
                    elif proc >= 2:
                        color = 'red'
                    elif proc >= 1:
                        color = 'yellow'
                    elif proc >= 0.5:
                        color = 'orange'
                    elif proc < 0.5:
                        color = 'black'
                    plt.plot([n3, n3], [min(k1, k2) + 0.05, max(k1, k2) - 0.05], color=color, markersize=1)
                    # live += 1
        n3 += 1

    x_predict1 = x[-1] + live
    x_predict2 = x2[-1] + live
    # print('x_predict1: {}->{}'.format(x[-1], live))
    # print('x_predict2: {}->{}'.format(x2[-1], live))

    plt.scatter(x, y, color='blue', marker=',')
    plt.scatter(x2, y2, color='red', marker=',')

    x = x[-len(x2):]
    y = y[-len(x2):]
    y, x = del_zerro_in_lean(y, x)
    x = np.asarray(x).reshape(len(x), 1)
    y = np.asarray(y).reshape(len(y), 1)
    regr.fit(x, y)

    x2 = x2[-len(x):]
    y2 = y2[-len(x):]
    y2, x2 = del_zerro_in_lean(y2, x2)
    y2 = np.asarray(y2).reshape(len(y2), 1)
    x2 = np.asarray(x2).reshape(len(x2), 1)
    regr2.fit(x2, y2)

    plt.plot(x, regr.predict(x) + get_std(y), color='blue', linestyle='dotted', markersize=1)
    plt.plot(x, regr.predict(x), color='black', linestyle='dashed', markersize=1)
    plt.plot(x, regr.predict(x) - get_std(y), color='blue', linestyle='dotted', markersize=1)

    plt.plot(x2, regr2.predict(x2) - get_std(y2), color='red', linestyle='dotted', markersize=1)
    plt.plot(x2, regr2.predict(x2), color='black', linestyle='dashed', markersize=1)
    plt.plot(x2, regr2.predict(x2) + get_std(y2), color='red', linestyle='dotted', markersize=1)

    kof_predict11 = round(float(regr.predict([[x_max]])[0]), 2)
    kof_predict21 = round(float(regr.predict([[x_predict1]])[0]), 2)

    kof_predict12 = round(float(regr2.predict([[x2_max]])[0]), 2)
    kof_predict22 = round(float(regr2.predict([[x_predict2]])[0]), 2)

    if kof_predict21 > kof_predict11:
        vect_fb = 'UP'
    elif kof_predict21 == kof_predict11:
        vect_fb = 'STAT'
    else:
        vect_fb = 'DOWN'
    print('Fonbet: {}, {}->{}. {}'.format(vect_fb, kof_predict11, kof_predict21, kof_cur1))

    if kof_predict22 > kof_predict12:
        vect_ol = 'UP'
    elif kof_predict22 == kof_predict12:
        vect_ol = 'STAT'
    else:
        vect_ol = 'DOWN'
    print('Olimp: {}, {}->{}. {}'.format(vect_ol, kof_predict12, kof_predict22, kof_cur2))
    plt.show()


def str_to_list_float(s: str) -> list:
    return list(map(float, s.replace('[', '').replace(']', '').replace(' ', '').split(',')))


def str_to_list_int(s: str) -> list:
    return list(map(int, s.replace('[', '').replace(']', '').replace(' ', '').split(',')))


if __name__ == '__main__':
    # x = [38, 90, 87, 90, 109, 69, 29, 88, 36, 31, 21, 14, 34, 9, 59, 31, 33, 9, 13, 41, 17, 11, 19, 26, 53, 19, 11, 3, 97, 9, 99, 33, 91, 24, 17, 101, 11, 5, 43, 31, 25, 1, 15, 3, 8, 28, 32, 29, 59, 2]
    # y = [2.45, 2.55, 2.5, 2.45, 2.4, 2.3, 2.35, 2.3, 2.25, 0, 2.25, 2.2, 2.3, 2.2, 2.15, 2.13, 2.1, 2.2, 2.18, 2.15, 2.18, 2.17, 2.18, 2.17, 2.15, 2.13, 2.05, 2.12, 2.1, 2.03, 2.1, 2.05, 2.0, 1.98, 1.97, 1.95, 1.93, 1.97, 0, 1.95, 1.93, 1.85, 1.83, 1.78, 1.92, 1.88, 1.95, 1.93, 1.9, 1.87]

    #    x2 = [56, 3, 3, 36, 34, 35, 127, 127, 95, 43, 33, 96, 79, 47, 15, 47, 48, 2, 75, 7, 87, 7, 55, 43, 95, 78, 1]
    #   y2 = [1.67, 1.7, 1.67, 1.7, 0, 1.7, 1.75, 1.8, 1.85, 1.9, 1.8, 1.85, 1.9, 1.85, 1.9, 1.8, 1.85, 1.8, 1.85, 1.9, 1.95, 2, 0, 2, 2.1, 2.15, 2.2]

    #  get_vect(x, y, x2, y2)

    df = pd.read_csv('D:\\YandexDisk\\Парсинг\\better\\06_05_2019_forks_simple.csv', encoding='utf-8', sep=';')
    df = df[df['l'] < 0.995]

    idx = df.groupby(['kof_ol', 'kof_fb', 'name'], sort=False)['live_fork_total'].transform('max') == df['live_fork_total']
    df = df[idx]

    for i, r in df.iterrows():
        x = str_to_list_int(r['fb_avg_change'])
        y = str_to_list_float(r['fb_kof_order'])
        x2 = str_to_list_int(r['ol_avg_change'])
        y2 = str_to_list_float(r['ol_kof_order'])
        lf = r['live_fork']
        lft = r['live_fork_total']
        l = (1 - r['l']) * 100
        print(x, y, x2, y2)
        print('proc: {}, t: {}, tt: {}, '.format(l, lf, lft))
        get_vect(x, y, x2, y2)
        print(''.ljust(150, '^'))
