"""
生成信号
"""
import pandas as pd
import numpy as np
# 简单移动平均线策略
def simple_moving_average_signal(df: pd.DataFrame,
                                 para: list = [20, 120]) -> pd.DataFrame:
    """
    简单的移动平均线策略。只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param df:
    :param para: ma_short, ma_long
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    ma_short = para[0]  # 短期均线。ma代表：moving_average
    ma_long = para[1]  # 长期均线

    # ===计算均线。所有的指标，都要使用复权价格进行计算。
    df['ma_short'] = df['close_fuquan'].rolling(ma_short, min_periods=1).mean()
    df['ma_long'] = df['close_fuquan'].rolling(ma_long, min_periods=1).mean()
    df['ma_short'] = np.around(df['ma_short'], decimals=3)
    df['ma_long'] = np.around(df['ma_long'], decimals=3)

    # ===找出做多信号
    condition1 = df['ma_short'] > df['ma_long']  # 短期均线 > 长期均线
    condition2 = df['ma_short'].shift(1) <= df['ma_long'].shift(
        1)  # 上一周期的短期均线 <= 长期均线
    df.loc[condition1 & condition2,
           'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['ma_short'] < df['ma_long']  # 短期均线 < 长期均线
    condition2 = df['ma_short'].shift(1) >= df['ma_long'].shift(
        1)  # 上一周期的短期均线 >= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    # ===删除无关中间变量
    df.drop(['ma_short', 'ma_long'], axis=1, inplace=True)
    return df