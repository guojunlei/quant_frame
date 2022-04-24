"""
多因子模型-选股回测框架
author:guojunlei
date:2022.04.08
"""
import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pandas as pd
import numpy as np
from tkdatabase.engine import FREngine, TimeseriesEngine
from decimal import Decimal, ROUND_HALF_UP

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)


def get_stock_code() -> np.array:
    '''
    获取A股股票代码列表
    :returns all_stocks:['000001',...'688688']
    '''
    engine = FREngine()._engine

    stock: pd.DataFrame = pd.read_sql_query(
        f"select symbol from security_info where exchange!='HK' ", con=engine)
    all_stock: np.array = np.unique(stock['symbol'].tolist())
    return all_stock


def import_index_data(date_end: str) -> pd.DataFrame:
    engine = TimeseriesEngine()._engine
    df: pd.DataFrame = pd.read_sql_query(
        f"select datetime,last from block_price_day where symbol='000001' and freq='1d'",
        con=engine,
        parse_dates=['datetime'])
    engine.dispose()
    df.sort_values(['datetime'], inplace=True)
    df['index_return'] = df['last'].pct_change()
    df = df[['datetime', 'index_return']]
    df.dropna(subset=['index_return'], inplace=True)
    df.rename(columns={'datetime': 'date'}, inplace=True)

    df = df[df['date'] <= pd.to_datetime(date_end)]
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x.date()))
    df.reset_index(inplace=True, drop=True)
    return df


def merge_with_index_data(df: pd.DataFrame,
                          index_data: pd.DataFrame) -> pd.DataFrame:
    # ===将股票数据和上证指数合并，结果已经排序
    df: pd.DataFrame = pd.merge(left=df,
                                right=index_data,
                                on='date',
                                how='right',
                                sort=True,
                                indicator=True)
    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['close'].fillna(method='ffill', inplace=True)
    # 用收盘价补全开盘价、最高价、最低价的空值
    df['close'].fillna(value=df['close'], inplace=True)
    df['high'].fillna(value=df['close'], inplace=True)
    df['low'].fillna(value=df['close'], inplace=True)
    # 补全前收盘价
    df['pre_close'].fillna(value=df['close'].shift(), inplace=True)
    # ===将停盘时间的某些列，数据填补为0
    fill_0_list = ['volume', 'money', 'return', 'open_return']
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)
    # ===去除上市之前的数据
    df = df[df['symbol'].notnull()]

    # ===判断计算当天是否交易
    df['is_trade'] = 1
    df.loc[df['_merge'] == 'right_only', 'is_trade'] = 0
    del df['_merge']

    df.reset_index(drop=True, inplace=True)
    return df


def get_one_stock_data(code: str) -> pd.DataFrame:
    engine = TimeseriesEngine()._engine
    df: pd.DataFrame = pd.read_sql_query(
        f"select * from stock_price_not_fuquan where symbol='{code}'",
        con=engine,
        parse_dates=['date'])
    engine.dispose()
    if df.empty:
        return pd.DataFrame()
    df = df[df['date'] <= pd.to_datetime(date_end)]
    if df.empty:
        return pd.DataFrame
    return df


# 计算涨跌停
def cal_zdt_price(df: pd.DataFrame) -> pd.DataFrame:
    # 计算涨停价格
    # 普通股票
    cond = df['is_st'] == 1
    df['limit_up'] = df['pre_close'] * 1.1
    df['limit_down'] = df['pre_close'] * 0.9
    df.loc[cond, 'limit_up'] = df['pre_close'] * 1.05
    df.loc[cond, 'limit_down'] = df['pre_close'] * 0.95

    # 2020年8月3日之后涨跌停规则有所改变
    # 新规的科创板
    new_rule_kcb = (df['date'] > pd.to_datetime('2020-08-03')
                    ) & df['symbol'].str.startswith('68')
    # 新规的创业板
    new_rule_cyb = (df['date'] > pd.to_datetime('2020-08-03')
                    ) & df['symbol'].str.startswith('30')

    # 科创板 & 创业板
    df.loc[new_rule_kcb | new_rule_cyb, 'limit_up'] = df['pre_close'] * 1.2
    df.loc[new_rule_kcb | new_rule_cyb, 'limit_down'] = df['pre_close'] * 0.8

    # 四舍五入
    df['limit_up'] = df['limit_up'].apply(lambda x: float(
        Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))
    df['limit_down'] = df['limit_down'].apply(lambda x: float(
        Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))

    # 判断是否一字涨停
    df['open_limit_up'] = False
    df.loc[df['low'] >= df['limit_up'], 'open_limit_up'] = True
    # 判断是否一字跌停
    df['open_limit_down'] = False
    df.loc[df['high'] <= df['limit_down'], 'open_limit_down'] = True
    # 判断是否开盘涨停
    df['open_up'] = False
    df.loc[df['open'] >= df['limit_up'], 'open_up'] = True
    # 判断是否开盘跌停
    df['open_down'] = False
    df.loc[df['open'] <= df['limit_down'], 'open_down'] = True

    return df


# 将日线数据转换为其他周期的数据
def transfer_to_period_data(df, period_type='m'):
    # 将交易日期设置为index
    df['last_date'] = df['date']
    df.set_index('date', inplace=True)
    period_df = df.resample(rule=period_type).agg({
        # 必须列
        'last_date': 'last',
        'symbol': 'last',
        'display_name': 'last',
        'is_trade': 'last',
        'is_st': 'last',
        'next_is_trade': 'last',
        'next_open_up': 'last',
        'next_is_st': 'last',
        'next_open_return': 'last',

        # 因子列
        'circulating_market_cap': 'last',
        'close': 'last',
    })

    # 计算必须额外数据
    period_df['trading_days'] = df['is_trade'].resample(period_type).sum()
    period_df['market_trading_days'] = df['symbol'].resample(
        period_type).size()
    period_df = period_df[period_df['market_trading_days'] >
                          0]  # 有的时候整个周期不交易（例如春节、国庆假期），需要将这一周期删除

    # 计算其他因子

    # 计算周期资金曲线
    period_df['every_day_return'] = df['return'].resample(period_type).apply(
        lambda x: list(x))

    # 重新设定index
    period_df.reset_index(inplace=True)
    period_df['date'] = period_df['last_date']
    del period_df['last_date']
    return period_df


def calculate_for_stock(code):
    print(code)
    df: pd.DataFrame = get_one_stock_data(code)
    if df.empty:
        return pd.DataFrame()

    # === 计算涨跌幅
    df['return'] = df['close'] / df['pre_close'] - 1
    df['open_return'] = df['close'] / df['open'] - 1  # 开盘买入的涨跌幅，为之后开盘买入用

    ################
    #这里需要计算所用到的factor.把函数写到factor.py中，传入df作为参数，返回pd.DataFrame
    ################
    ##---------------------------------------------------

    ##---------------------------------------------------

    # =将股票和上证指数合并，补全停牌的日期，新增数据"is_trade"、"index_return"
    df = merge_with_index_data(df, index_data)
    if df.empty:
        return pd.DataFrame()

    # =计算涨跌停价格
    df = cal_zdt_price(df)

    # =计算下个交易的相关情况
    df['next_is_trade'] = df['is_trade'].shift(-1)
    df['next_open_limit_up'] = df['open_limit_up'].shift(-1)
    df['next_open_up'] = df['open_up'].shift(-1)
    df['next_is_st'] = (df['is_st'] == 1).shift(-1)
    df['next_open_return'] = df['open_return'].shift(-1)

    # =将日线数据转化为月线或者周线
    df = transfer_to_period_data(df, period_type=period_type)

    # 删除2017年之前的数据
    df = df[df['date'] > pd.to_datetime(date_start)]
    # 计算下周期每天涨幅
    if df.empty:
        return pd.DataFrame
    df['next_every_day_return'] = df['every_day_return'].shift(-1)
    del df['every_day_return']
    
    # =删除不能交易的周期数
    # 删除月末为st状态的周期数
    df = df[df['is_st'] != 1]

    # 删除月末不交易的周期数
    df = df[df['is_trade'] == 1]
    # 删除交易天数过少的周期数
    df = df[df['trading_days'] / df['market_trading_days'] >= 0.8]
    df.drop(['trading_days', 'market_trading_days'], axis=1, inplace=True)
    return df


if __name__ == '__main__':

    period_type = 'M'  # W代表周，M代表月
    date_start = '2007-01-01'  # 回测开始时间
    date_end = '2022-04-14'  # 回测结束时间

    # ===读取所有股票代码的列表
    stock_code_list = get_stock_code()

    # 导入上证指数，保证指数数据和股票数据在同一天结束，不然会出现问题。
    index_data = import_index_data(date_end)

    df = calculate_for_stock('300090')
    exit()
    start_time = datetime.datetime.now()

    with Pool(max(cpu_count() - 2, 1)) as pool:
        df_list = pool.map(calculate_for_stock, sorted(stock_code_list))

    print('读入完成, 开始合并，消耗事件', datetime.datetime.now() - start_time)

    # 合并为一个大的DataFrame
    all_stock_data = pd.concat(df_list, ignore_index=True)
    all_stock_data.sort_values(['date', 'symbol'], inplace=True)
    all_stock_data.reset_index(inplace=True, drop=True)

    all_stock_data.to_parquet("data/all_stock_data.parquet")
    print(datetime.datetime.now() - start_time)