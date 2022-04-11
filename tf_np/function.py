"""
timing_frame 下文件所用的到函数
"""
from unicodedata import decimal
import numpy as np
import sqlite3 as sql
import datetime
from decimal import Decimal, ROUND_HALF_UP
import numba as nb
"""
整理数据所需函数
"""


def get_one_stock_data(code: str) -> np.array:
    conn = sql.connect('../../data/All_stock.db')
    cur = conn.cursor()

    # ===获取单只股票数据
    _sql: str = f"select * from stock where symbol = '{code}' "
    rs = cur.execute(_sql)
    data: list = rs.fetchall()

    # === 获取所有columns名
    _sql: str = "PRAGMA table_info(stock)"
    rs = cur.execute(_sql)
    col: list = rs.fetchall()
    col_name: list = [c[1] for c in col]

    fmt: list = [
        'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
        'float64', 'U20', 'U20', 'U10', 'U10', 'U20', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'float64'
    ]
    stock_type: list = list(zip(col_name, fmt))
    stock: np.array = np.array(data, dtype=stock_type)
    return stock


def cal_fuquan_price(arr: np.array) -> np.array:
    factor: np.array = np.cumprod(arr[3] / arr[4])  # 复权因子
    close_f: np.array = factor * (arr[3][-1] / factor[-1])  #前复权
    open_f: np.array = arr[0] / arr[3] * close_f
    high_f: np.array = arr[1] / arr[3] * close_f
    l_f: np.array = arr[2] / arr[3] * close_f
    fuquan: np.array = np.array([open_f, high_f, l_f, close_f])
    return fuquan


def cal_limit_price(arr: np.array) -> np.array:
    limit_up: np.array = arr[2].astype('float64') * 1.1
    limit_down: np.array = arr[2].astype('float64') * 0.9
    is_st = arr[1].astype('float64')
    cond = is_st == 1
    limit_up[cond] = arr[2][cond].astype('float64') * 1.05
    limit_down[cond] = arr[2][cond].astype('float64') * 0.95

    # ===科创板,创业板
    mask_kcb_1 = []
    for i, v in enumerate(arr[3]):
        if v.startswith('68') | v.startswith('30'):
            mask_kcb_1.append(i)

    mask_kcb_2 = []
    for i, v in enumerate(arr[0]):
        v = datetime.datetime.strptime(v.split(' ')[0], "%Y-%m-%d")
        s_time = datetime.datetime(2020, 8, 3)
        if v > s_time:
            mask_kcb_2.append(i)
    mask = list(np.intersect1d(mask_kcb_1, mask_kcb_2))

    limit_up[mask] = arr[2][mask].astype('float64') * 1.2
    limit_down[mask] = arr[2][mask].astype('float64') * 0.8

    up: list = []
    for v in limit_up:
        v: float = float(
            Decimal(v * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) /
            100)
        up.append(v)
    down: list = []
    for v in limit_down:
        v: float = float(
            Decimal(v * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) /
            100)
        down.append(v)
    return np.array([np.array(up), np.array(down)])


def cal_is_to_limit(arr, limit: np.array) -> np.array:
    is_up_down: np.array = np.zeros(shape=[4, limit.shape[1]])

    cond = arr[1] >= limit[0]
    is_up_down[0][cond] = 1

    cond = arr[0] <= limit[1]
    is_up_down[1][cond] = 1

    cond = arr[2] >= limit[0]
    is_up_down[2][cond] = 1

    cond = arr[2] <= limit[1]
    is_up_down[3][cond] = 1

    return is_up_down

def fill_na(arr: np.array, method='ffill', types='float') -> np.array:
    if types == 'float':
        if method == 'ffill':
            na_arr: list = []
            for i, v in enumerate(arr):
                if np.isnan(v):
                    if i == 0:
                        na_arr.append(np.nan)
                        continue
                    else:
                        j: int = i - 1
                        while j >= 0:
                            if not np.isnan(arr[j]):
                                na_arr.append(arr[j])
                                break
                            j -= 1
                        if j == -1:
                            na_arr.append(np.nan)
                            continue
                else:
                    na_arr.append(v)
            return np.array(na_arr)
    elif types == 'str':
        if method == 'ffill':
            na_arr: list = []
            for i, v in enumerate(arr):
                if len(v) == 3:
                    if i == 0:
                        na_arr.append(np.nan)
                        continue
                    else:
                        j: int = i - 1
                        while j >= 0:
                            if len(arr[j]) != 3:
                                na_arr.append(arr[j])
                                break
                            j -= 1
                        if j == -1:
                            na_arr.append(np.nan)
                            continue
                else:
                    na_arr.append(v)
            return np.array(na_arr)


"""
计算signal所需函数
"""


# === 简单的移动平均线策略（需要两个参数，长短周期）
def simple_moving_average_signal(arr: np.array, para: list) -> np.array:
    """
    简单的移动平均线策略。只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param arr:
    :param para: ma_short, ma_long
    :return: np.array signal信号
    """
    ma_short: int = para[0]
    ma_long: int = para[1]

    # === 计算均线（需要复权价）
    arr_short: list = []
    arr_long: list = []

    for i, v in enumerate(arr):
        i += 1
        if i < ma_short:
            arr_short.append(np.mean(arr[:i]))
        else:
            arr_short.append(np.mean(arr[i - ma_short:i]))

        if i < ma_long:
            arr_long.append(np.mean(arr[:i]))
        else:
            arr_long.append(np.mean(arr[i - ma_long:i]))
    _short = [np.nan] + arr_short[:-1]
    _long = [np.nan] + arr_long[:-1]

    arr_short: np.array = np.array(arr_short)
    arr_long: np.array = np.array(arr_long)
    _short: np.array = np.array(_short)
    _long: np.array = np.array(_long)
    # 做多
    cond = (arr_short > arr_long) & (_short <= _long)
    signal: np.array = np.full_like(arr_short, np.nan)
    signal[cond] = 1
   
    #平仓
    cond = (arr_short < arr_long) & (_short >= _long)
    signal[cond] = 0
    # signal: np.array = fill_na(signal, method='ffill', types='float')
    return signal


"""
由signal产生实际持仓
"""


def position_at_close(signal, close, up_down: np.array) -> np.array:
    """
    根据signal产生实际持仓。考虑涨跌停不能买入卖出的情况。
    所有的交易都是发生在产生信号的K线的结束时
    """
    # ===由signal计算出实际的每天持有仓位
    # 在产生signal的k线结束的时候，进行买入
    signal: np.array = fill_na(signal, method='ffill', types='float')
    signal[np.isnan(signal)] = 0
    pos: np.array = np.insert(signal[:-1], 0, [0])

    # ===对涨跌停无法买卖做出相关处理。

    # 找出收盘价无法买入的K线
    cond1 = close >= up_down[0]
    cond2 = pos == 1
    pos[(cond1 & cond2)] = np.nan

    # 找出收盘价无法卖出的K线
    cond1 = close <= up_down[1]
    cond2 = pos == 0
    pos[(cond1 & cond2)] = np.nan
    pos: np.array = fill_na(pos, method='ffill', types='float')
    return pos


"""
计算资金曲线
"""

def equity_curve_with_long_at_close(pos: np.array,
                                    close_fuquan: np.array,
                                    pre_close: np.array,
                                    close: np.array,
                                    arr_date: np.array,
                                    c_rate: np.float64 = 2.5 / 10000,
                                    t_rate: np.float64 = 1.0 / 1000,
                                    slippage: np.float64 = 0.01) -> np.array:
    """
    计算股票的资金曲线。只能做多，不能做空。并且只针对满仓操作
    每次交易是以当根K线的收盘价为准。
    :param c_rate: 手续费，commission fees，默认为万分之2.5
    :param t_rate: 印花税，tax，默认为千分之1。etf没有
    :param slippage: 滑点，股票默认为0.01元，etf为0.001元
    """
    # ==找出开仓、平仓条件
    cond1 = pos != 0
    _pos = np.insert(pos[:-1], 0, np.nan)
    cond2 = _pos != pos
    cond_open = cond1 & cond2

    cond1 = pos != 0
    _pos = np.append(pos[1:], [np.nan])
    cond2 = _pos != pos
    cond_close = cond1 & cond2

    index_open = np.where(cond_open)
    index_close = np.where(cond_close)

    start_time: np.array = np.full_like(pos, np.nan, dtype='U20')
    start_time[index_open] = arr_date[index_open]
    start_time = fill_na(start_time, method='ffill', types='str')
    start_time[pos == 0] = np.nan

    # 初始资金，默认为1000000元
    initial_cash: int = 100000

    # ===在买入的K线
    # 在发出信号的当根K线以收盘价买入 实际买入股票数量
    stock_num: np.array = np.full_like(pos, np.nan, dtype=np.float64)
    stock_num[index_open] = np.floor(
        (initial_cash * (1 - c_rate) /
         (pre_close[index_open] + slippage)) / 100) * 100

    # 买入股票之后剩余的钱，扣除了手续费
    cash: np.array = np.full_like(pos, np.nan, dtype=np.float64)
    cash[~np.
         isnan(stock_num)] = initial_cash - stock_num[~np.isnan(stock_num)] * (
             pre_close[~np.isnan(stock_num)] + slippage) * (1 + c_rate)

    stock_value: np.array = np.full_like(pos, np.nan, dtype=np.float64)
    stock_value[~np.isnan(stock_num)] = stock_num[
        ~np.isnan(stock_num)] * close[~np.isnan(stock_num)]

    # ===在买入之后的K线
    # 买入之后现金不再发生变动
    cash = fill_na(cash, method='ffill', types='float')
    cash[pos == 0] = np.nan

    # 股票净值随着涨跌幅波动
    group_list: list = []
    _g: list = []
    for i, v in enumerate(cash):
        if i == 0:
            continue
        elif not np.isnan(v):
            _g.append(i)
        elif np.isnan(v) & ~np.isnan(cash[i - 1]):
            group_list.append(_g)
            _g: list = []
    group_num: int = len(group_list)
    # print(group_num)
    # exit()

    if group_num > 1:
        for g in group_list:
            stock_value[
                g] = close_fuquan[g] / close_fuquan[g][0] * stock_value[g][0]
        
    elif group_num == 1:
        stock_value[group_list[0]] / close_fuquan[
            group_list[0]][0] * stock_value[group_list[0]][0]
   
    # ===在卖出的K线
    # 股票数量变动
    stock_num[index_close] = stock_value[index_close] / close[index_close]
    
    cash[index_close] += stock_num[index_close] * (
        close[index_close] - slippage) * (1 - c_rate - t_rate)
   
    stock_value[index_close] = 0

    net_value: np.array = stock_value + cash
  
    # ===计算资金曲线
    equity_change: np.array = (net_value - np.insert(net_value[:-1], 0, np.nan)) / np.insert(net_value[:-1], 0, np.nan)
    
    equity_change[index_open] = net_value[index_open] / initial_cash - 1
    equity_change[np.isnan(equity_change)] = 0
    equity_change=np.around(equity_change,decimals=4)
    equity_curve: np.array = np.cumprod(1 + equity_change)
    
    equity_curve_base = np.cumprod(close / pre_close)
    return equity_curve, equity_curve_base
