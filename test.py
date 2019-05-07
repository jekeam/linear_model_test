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
def del_zerro_in_lean(val_arr: list, line_arr: list):
    if len(val_arr) > len(line_arr):
        val_arr = val_arr[-len(line_arr):]
    elif len(line_arr) > len(val_arr):
        line_arr = line_arr[-len(val_arr):]
    for idx, val in enumerate(val_arr):
        if val == 0:
            val_arr.pop(idx)
            line_arr.pop(idx)
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
    # x = [30, 89, 7, 20, 29, 59, 29, 59, 28, 59, 29, 29, 29, 59, 28, 25, 33, 29, 29, 29, 59, 60, 27, 87, 11, 23, 54, 59, 91, 56, 59, 59, 89, 58, 61, 59, 57, 88, 49, 61, 77, 80, 29, 29,
    #      59,
    #      29, 12, 75, 53, 141, 17, 25, 1133, 29, 29, 29, 29, 58, 59, 59, 29, 59, 29, 28, 29, 59, 29, 41, 49, 29, 58, 29, 89, 29, 29, 57, 28, 31, 57, 15, 83, 19, 29, 17, 15, 46, 37, 29,
    #      59,
    #      31, 27, 29, 31, 7, 18, 59, 29, 21, 7, 29, 31, 27, 31, 27, 29, 31, 27, 26, 23, 37, 29, 31, 27, 25, 33, 29, 7, 21, 31, 50]
    # y = [4.4, 4.3, 4.2, 4.15, 4.1, 4.05, 4.0, 3.95, 3.9, 3.85, 3.8, 3.75, 3.7, 3.65, 3.6, 3.55, 3.5, 3.45, 3.4, 3.35, 3.3, 3.25, 3.2, 3.15, 3.1, 3.0, 2.95, 2.9, 2.85, 2.8, 2.75, 2.7,
    #      2.65,
    #      2.6, 2.55, 2.5, 2.45, 2.4, 2.35, 2.3, 0, 2.2, 2.15, 2.13, 2.1, 2.08, 2.05, 2.02, 2.0, 0, 1.87, 1.85, 1.8, 1.78, 1.77, 1.75, 1.73, 1.72, 1.68, 1.65, 1.64, 1.62, 1.6, 1.58, 1.57,
    #      1.55, 1.52, 1.5, 1.48, 1.47, 1.45, 1.42, 1.4, 1.38, 1.37, 1.35, 1.33, 1.32, 1.3, 1.28, 0, 2.5, 2.45, 2.4, 0, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.08, 2.05, 2.02, 1.93, 1.9, 1.85,
    #      1.83, 1.88, 1.87, 1.83, 1.8, 1.75, 1.72, 1.68, 1.67, 1.65, 1.62, 1.6, 1.48, 1.47, 1.45, 1.43, 1.4, 1.38, 1.35, 1.33, 1.35, 1.33, 1.3]
    #
    # x2 = [19, 102, 159, 11, 23, 102, 91, 67, 27, 31, 19, 31, 966, 86, 55, 15, 47, 123, 53, 49, 55, 15, 67, 11, 15, 55, 76, 46, 79, 35, 31, 39, 11, 3, 47, 27, 11, 19, 51, 32, 51, 27, 7,
    #       11,
    #       11, 24, 35, 51, 47, 19, 46, 31, 47, 19, 40, 1]
    # y2 = [1.6, 1.65, 1.7, 1.75, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 1.9, 1.95, 1.9, 1.95, 2.02, 1.95, 2.02, 2.1, 2.25, 2.3, 0, 1.57, 1.6, 1.65, 0, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    #       2.02,
    #       1.95, 2.02, 2.1, 2.02, 2.1, 2.2, 2.25, 2.3, 2.4, 2.5, 2.4, 2.5, 2.55, 2.65, 2.7, 2.95, 3.05, 3.2, 3.4, 3.57, 3.83, 4.05, 4.42]

    df = pd.read_csv('D:\\YandexDisk\\Парсинг\\better\\06_05_2019_forks_simple.csv', encoding='utf-8', sep=';')
    df = df[df['l'] < 0.995]

    idx = df.groupby(['kof_ol', 'kof_fb', 'name'], sort=False)['live_fork_total'].transform('max') == df['live_fork_total']
    df = df[idx]

    for i, r in df.iterrows():
        x = str_to_list_int(r['fb_avg_change'])
        y = str_to_list_float(r['fb_kof_order'])
        x2 = str_to_list_int(r['ol_avg_change'])
        y2 = str_to_list_float(r['ol_kof_order'])
        print(x, y, x2, y2)
        print(''.ljust(150, '_'))
        get_vect(x, y, x2, y2)
        print(''.ljust(150, '^'))
