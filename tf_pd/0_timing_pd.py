"""
author: guojunlei
择时框架
2022-04-08
"""
from function import *
import time

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)

s_time = time.time()
# 读入股票数据
assets_name = '600000'
df = get_one_stock_data(assets_name)
df.sort_values(by=['date'], inplace=True)
df.drop_duplicates(subset=['date'], inplace=True)
df.reset_index(inplace=True, drop=True)

# 计算复权价格、涨停价格
df = cal_fuquan_price(df, fuquan_type='前复权')
df = cal_zdt_price(df)

# 参数
para = [20, 30]

# 计算交易信号
df = simple_moving_average_signal(df, para=para)

# 计算实际持仓
df = position_at_close(df)

# 选择时间段
# 截取上市一年之后的数据
# df = df.iloc[250 - 1:]  # 股市一年交易日大约250天
# # 截图2007年之后的数据
# df = df[df['交易日期'] >= pd.to_datetime('20070101')]

df = equity_curve_with_long_at_close(df,
                                     c_rate=1.5 / 10000,
                                     t_rate=1.0 / 1000,
                                     slippage=0.01)

equity_curve = df.iloc[-1]['equity_curve']
equity_curve_base = df.iloc[-1]['equity_curve_base']
print(para, '策略最终收益：', equity_curve)
print(time.time() - s_time)
