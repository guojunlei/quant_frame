"""
计算signal所需函数
"""
import numpy as np

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