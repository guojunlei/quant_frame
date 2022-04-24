import pandas as pd
import sqlite3 as sql
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
"""
整理数据所需函数
"""


# === 获取单只股票数据
def get_one_stock_data(code: str) -> pd.DataFrame:
    conn = sql.connect('../data/All_stock.db')
    df: pd.DataFrame = pd.read_sql_query(
        f"select * from stock where symbol='{code}' ",
        con=conn,
        parse_dates=['date'])
    conn.close()
    return df


# =====计算复权价格
def cal_fuquan_price(df: pd.DataFrame, fuquan_type='后复权') -> pd.DataFrame:
    """
    用于计算复权价格
    :param fuquan_type: ‘前复权’或者‘后复权’
    :return: 最终输出的df中，新增字段：收盘价_复权，开盘价_复权，最高价_复权，最低价_复权
    """

    # 计算复权因子
    df['复权因子'] = (df['close'] / df['pre_close']).cumprod()

    # 计算前复权、后复权收盘价
    if fuquan_type == '后复权':
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    elif fuquan_type == '前复权':
        df['close_fuquan'] = df['复权因子'] * (df.iloc[-1]['close'] /
                                           df.iloc[-1]['复权因子'])
    else:
        raise ValueError('计算复权价时，出现未知的复权类型：%s' % fuquan_type)

    # 计算复权
    df['open_fuquan'] = df['open'] / df['close'] * df['close_fuquan']
    df['high_fuquan'] = df['high'] / df['close'] * df['close_fuquan']
    df['low_fuquan'] = df['low'] / df['close'] * df['close_fuquan']
    df.drop(['复权因子'], axis=1, inplace=True)

    return df


# 计算涨跌停
def cal_zdt_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含st股，但是不包含新股
    涨跌停制度规则:
        ---2020年8月3日
        非ST股票 10%
        ST股票 5%

        ---2020年8月4日至今
        普通非ST股票 10%
        普通ST股票 5%

        科创板（sh68） 20%
        创业板（sz30） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    :return:
    """
    # 计算涨停价格
    # 普通股票
    cond = df['is_st'] == 1
    df['up_limit'] = df['pre_close'] * 1.1
    df['down_limit'] = df['pre_close'] * 0.9
    df.loc[cond, 'up_limit'] = df['pre_close'] * 1.05
    df.loc[cond, 'down_limit'] = df['pre_close'] * 0.95

    # 2020年8月3日之后涨跌停规则有所改变
    # 新规的科创板
    new_rule_kcb = (df['date'] > pd.to_datetime('2020-08-03')
                    ) & df['symbol'].str.startswith('68')
    # 新规的创业板
    new_rule_cyb = (df['date'] > pd.to_datetime('2020-08-03')
                    ) & df['symbol'].str.startswith('30')
    # # 北交所条件
    # cond_bj = df['股票代码'].str.contains('bj')

    # 科创板 & 创业板
    df.loc[new_rule_kcb | new_rule_cyb, 'up_limit'] = df['pre_close'] * 1.2
    df.loc[new_rule_kcb | new_rule_cyb, 'down_limit'] = df['pre_close'] * 0.8

    # 北交所
    # df.loc[cond_bj, '涨停价'] = df['前收盘价'] * 1.3
    # df.loc[cond_bj, '跌停价'] = df['前收盘价'] * 0.7

    # 四舍五入
    df['up_limit'] = df['up_limit'].apply(lambda x: float(
        Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))
    df['down_limit'] = df['down_limit'].apply(lambda x: float(
        Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))

    # 判断是否一字涨停
    df['一字涨停'] = False
    df.loc[df['open'] >= df['up_limit'], '一字涨停'] = True
    # 判断是否一字跌停
    df['一字跌停'] = False
    df.loc[df['high'] <= df['down_limit'], '一字跌停'] = True
    # 判断是否开盘涨停
    df['开盘涨停'] = False
    df.loc[df['open'] >= df['up_limit'], '开盘涨停'] = True
    # 判断是否开盘跌停
    df['开盘跌停'] = False
    df.loc[df['open'] <= df['down_limit'], '开盘跌停'] = True

    return df


# 由交易信号产生实际持仓
def position_at_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据signal产生实际持仓。考虑涨跌停不能买入卖出的情况。
    所有的交易都是发生在产生信号的K线的结束时
    :param df:
    :return:
    """
    # ===由signal计算出实际的每天持有仓位
    # 在产生signal的k线结束的时候，进行买入
    df['signal'].fillna(method='ffill', inplace=True)
    df['signal'].fillna(value=0, inplace=True)  # 将初始行数的signal补全为0
    df['pos'] = df['signal'].shift()
    df['pos'].fillna(value=0, inplace=True)  # 将初始行数的pos补全为0

    # ===对涨跌停无法买卖做出相关处理。
    # 找出收盘价无法买入的K线
    cannot_buy_condition = df['close'] >= df['up_limit']
    # 将找出上一周期无法买入的K线、并且signal为1时，的'pos'设置为空值
    df.loc[cannot_buy_condition.shift() & (df['signal'].shift() == 1),
           'pos'] = None  # 2010-12-22

    # 找出收盘价无法卖出的K线
    cannot_sell_condition = df['close'] <= df['down_limit']
    # 将找出上一周期无法卖出的K线、并且signal为0时的'pos'设置为空值
    df.loc[cannot_sell_condition.shift() & (df['signal'].shift() == 0),
           'pos'] = None

    # pos为空的时，不能买卖，只能和前一周期保持一致。
    df['pos'].fillna(method='ffill', inplace=True)

    # ===如果是分钟级别的数据，还需要对t+1交易制度进行处理，在之后案例中进行演示

    # ===删除无关中间变量
    df.drop(['signal'], axis=1, inplace=True)
    return df


# =====计算资金曲线
# 股票资金曲线
def equity_curve_with_long_at_close(df,
                                    c_rate=2.5 / 10000,
                                    t_rate=1.0 / 1000,
                                    slippage=0.01):
    """
    计算股票的资金曲线。只能做多，不能做空。并且只针对满仓操作
    每次交易是以当根K线的收盘价为准。
    :param df:
    :param c_rate: 手续费，commission fees，默认为万分之2.5
    :param t_rate: 印花税，tax，默认为千分之1。etf没有
    :param slippage: 滑点，股票默认为0.01元，etf为0.001元
    :return:
    """

    # ==找出开仓、平仓条件
    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(1)
    open_pos_condition = condition1 & condition2

    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(-1)
    close_pos_condition = condition1 & condition2

    # ==对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['date']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # ===基本参数
    initial_cash = 100000  # 初始资金，默认为1000000元

    # ===在买入的K线
    # 在发出信号的当根K线以收盘价买入
    df.loc[open_pos_condition,
           'stock_num'] = initial_cash * (1 - c_rate) / (df['pre_close'] +
                                                         slippage)

    # 实际买入股票数量
    df['stock_num'] = np.floor(df['stock_num'] / 100) * 100

    # 买入股票之后剩余的钱，扣除了手续费
    df['cash'] = initial_cash - df['stock_num'] * (df['pre_close'] +
                                                   slippage) * (1 + c_rate)

    # 收盘时的股票净值
    df['stock_value'] = df['stock_num'] * df['close']

    # ===在买入之后的K线
    # 买入之后现金不再发生变动
    df['cash'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, ['cash']] = None

    # 股票净值随着涨跌幅波动
    group_num = len(df.groupby('start_time'))

    if group_num > 1:
        t = df.groupby('start_time').apply(
            lambda x: x['close_fuquan'] / x.iloc[0]['close_fuquan'] * x.iloc[
                0]['stock_value'])
        t = t.reset_index(level=[0])
        df['stock_value'] = t['close_fuquan']
    elif group_num == 1:
        t = df.groupby('start_time')[[
            'close_fquan', 'stock_value'
        ]].apply(lambda x: x['close_fuquan'] / x.iloc[0]['close_fuquan'] * x.
                 iloc[0]['stock_value'])
        df['stock_value'] = t.T.iloc[:, 0]

    # ===在卖出的K线
    # 股票数量变动
    df.loc[close_pos_condition,
           'stock_num'] = df['stock_value'] / df['close']  # 看2006年初
    # 现金变动
    df.loc[close_pos_condition,
           'cash'] += df.loc[close_pos_condition, 'stock_num'] * (
               df['close'] - slippage) * (1 - c_rate - t_rate)

    # 股票价值变动
    df.loc[close_pos_condition, 'stock_value'] = 0

    # ===账户净值
    df['net_value'] = df['stock_value'] + df['cash']

    # ===计算资金曲线
    df['equity_change'] = df['net_value'].pct_change(fill_method=None)
    df.loc[open_pos_condition,
           'equity_change'] = df.loc[open_pos_condition,
                                     'net_value'] / initial_cash - 1  # 开仓日的收益率
    df['equity_change'].fillna(value=0, inplace=True)
    df['equity_change'] = np.around(df['equity_change'], decimals=4)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()
    df['equity_curve_base'] = (df['close'] / df['pre_close']).cumprod()

    # ===删除无关数据
    df.drop(['start_time', 'stock_num', 'cash', 'stock_value', 'net_value'],
            axis=1,
            inplace=True)
    return df
