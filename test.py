import numpy as np
# from sklearn.svm import SVR
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy


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
def del_zerro_in_line(val_arr: list, line_arr: list):
    print('start')
    print(len(val_arr), len(line_arr), val_arr)
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
    print(len(list(filter(lambda x: x != 0, val_arr))))
    print(len(val_arr), len(line_arr), val_arr)
    print('end')
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
    y, x = del_zerro_in_line(y, x)
    x = np.asarray(x).reshape(len(x), 1)
    y = np.asarray(y).reshape(len(y), 1)
    regr.fit(x, y)

    x2 = x2[-len(x):]
    y2 = y2[-len(x):]
    y2, x2 = del_zerro_in_line(y2, x2)
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
    is_id = None
    is_id = [1557322506,
             1557352676,
             1557321737,
             1557329583,
             1557327925,
             1557341977,
             1557332068, ]
    with open('C:\\Users\\User\\Documents\\GitHub\\linear_model_test\\09_05_2019_11_id_forks.txt', encoding='utf-8') as f:
        fl = f.readlines()

    min_profit = 0
    max_profit = 0
    sale_profit = 0

    for r in fl:
        r = json.loads(r.strip())
        for id, r in r.items():
            if (is_id and int(id) in is_id) or is_id is None:
                x = str_to_list_int(r.get('fonbet', {})['avg_change'])
                y = str_to_list_float(r.get('fonbet', {})['order_kof'])
                x2 = str_to_list_int(r.get('olimp', {})['avg_change'])
                y2 = str_to_list_float(r.get('olimp', {})['order_kof'])

                bk1 = r.get('olimp')
                bk2 = r.get('fonbet')
                err_bk1, err_bk2 = bk1.get('err'), bk2.get('err')
                bet_skip = False

                if err_bk1 and err_bk2:
                    if 'BkOppBetError' in err_bk1 and 'BkOppBetError' in err_bk2:
                        bet_skip = True

                if err_bk1 != 'ok' or err_bk2 != 'ok':
                    if not bet_skip:
                        sale_profit = sale_profit + bk1.get('sale_profit') + bk2.get('sale_profit')

                elif not bet_skip:
                    sum_bet1, sum_bet2 = bk1.get('new_bet_sum'), bk2.get('new_bet_sum')
                    k1, k2 = bk1.get('new_bet_kof'), bk2.get('new_bet_kof')
                    if sum_bet1 and sum_bet2 and k1 and k2:
                        total_sum = sum_bet1 + sum_bet2
                        min_profit = min_profit + round(min((sum_bet1 * k1 - total_sum), (sum_bet2 * k2 - total_sum)))
                        max_profit = max_profit + round(max((sum_bet1 * k1 - total_sum), (sum_bet2 * k2 - total_sum)))

                res_str = str(id) + ': '
                res_str = res_str + 'min профит: ' + '{:,}'.format(round(min_profit)).replace(',', ' ') + ', '
                res_str = res_str + 'max профит: ' + '{:,}'.format(round(max_profit)).replace(',', ' ') + ', '
                res_str = res_str + 'Выкупы: ' + '{:,}'.format(round(sale_profit)).replace(',', ' ') + ', '
                res_str = res_str + '~ доход: ' + '{:,}'.format(round((max_profit + min_profit) / 2) + round(sale_profit)).replace(',', ' ')
                print(res_str)
                get_vect(x, y, x2, y2)
                print(''.rjust(150, '-'))

    # df = pd.read_csv('D:\\YandexDisk\\Парсинг\\better\\logs\\07.05.19\\08_05_2019_forks_simple.csv', encoding='utf-8', sep=';')
    # df = df[df['l'] < 0.995]
    #
    # idx = df.groupby(['kof_ol', 'kof_fb', 'name'], sort=False)['live_fork_total'].transform('max') == df['live_fork_total']
    # df = df[idx]
    #
    # for i, r in df.iterrows():
    #     x = str_to_list_int(r['fb_avg_change'])
    #     y = str_to_list_float(r['fb_kof_order'])
    #     x2 = str_to_list_int(r['ol_avg_change'])
    #     y2 = str_to_list_float(r['ol_kof_order'])
    #     lf = r['live_fork']
    #     lft = r['live_fork_total']
    #     l = (1 - r['l']) * 100
    #     print(x, y, x2, y2)
    #     print('proc: {}, t: {}, tt: {}, '.format(l, lf, lft))
    #     get_vect(x, y, x2, y2)
    #     print(''.ljust(150, '^'))
