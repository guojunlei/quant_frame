"""
author: guojunlei
择时框架
2022-04-08
"""
from function import *
import time
from signals import *

np.set_printoptions(threshold=np.inf)

s_time = time.time()
# === 读入单只标的数据
assets_name = '600000'
assets = get_one_stock_data(assets_name)

# 计算复权价格、涨停价格 (np.array([open_f, high_f, l_f, close_f]))
fuquan_price: np.array = cal_fuquan_price(
    np.array([
        assets['open'], assets['high'], assets['low'], assets['close'],
        assets['pre_close']
    ]))

daily_limit: np.array = cal_limit_price(
    np.array([
        assets['date'], assets['is_st'], assets['pre_close'], assets['symbol']
    ]))

is_to_limit: np.array = cal_is_to_limit(
    np.array([assets['high'], assets['low'], assets['open']]), daily_limit)

# === 设置指标参数
para = [20, 30]

# === 计算交易信号 （需要在function中单独写）
signal: np.array = simple_moving_average_signal(fuquan_price[3], para)

# === 计算实际持仓
pos: np.array = position_at_close(signal, assets['close'], daily_limit)

# === 选择时间段
# 截取上市一年之后的数据
ipo_days: int = 250

# === 计算资金曲线
equity, base = equity_curve_with_long_at_close(pos,
                                               fuquan_price[-1],
                                               assets['pre_close'],
                                               assets['close'],
                                               assets['date'],
                                               c_rate=1.5 / 10000,
                                               t_rate=1 / 1000,
                                               slippage=0.01)
print(time.time() - s_time)
print(equity[-1])