"""
整理所有股票数据后，用于策略选股，生成资金曲线
author:guojunlei
date:2022.4.8
"""
import pandas as pd
import numpy as np
from tkdatabase.engine import FREngine, TimeseriesEngine

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)


def import_index_data(back_trader_start=None,
                      back_trader_end=None) -> pd.DataFrame:
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
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x.date()))
    if back_trader_start:
        df = df[df['date'] >= pd.to_datetime(back_trader_start)]
    if back_trader_end:
        df = df[df['date'] <= pd.to_datetime(back_trader_end)]

    df.reset_index(inplace=True, drop=True)
    return df


def create_empty_data(index_data, period):
    empty_df = index_data[['date']].copy()
    empty_df['return'] = 0.0
    empty_df['last_date'] = empty_df['date']
    empty_df.set_index('date', inplace=True)
    agg_dict = {'last_date': 'last'}
    empty_period_df = empty_df.resample(rule=period).agg(agg_dict)

    empty_period_df['every_day_return'] = empty_df['return'].resample(
        period).apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['last_date'], inplace=True)

    empty_period_df['next_every_day_return'] = empty_period_df[
        'every_day_return'].shift(-1)
    empty_period_df.dropna(subset=['next_every_day_return'], inplace=True)

    # 填仓其他列
    empty_period_df['stock_num'] = 0
    empty_period_df['buy_stock_code'] = 'empty'
    empty_period_df['buy_stock_name'] = 'empty'
    empty_period_df['next_return'] = 0.0
    empty_period_df.rename(columns={'last_date': 'date'}, inplace=True)
    empty_period_df.set_index('date', inplace=True)

    empty_period_df = empty_period_df[[
        'stock_num', 'buy_stock_code', 'buy_stock_name', 'next_return',
        'next_every_day_return'
    ]]
    return empty_period_df


if __name__ == '__main__':
    # ===参数设定
    select_stock_num = 3  # 选股数量
    c_rate = 1.5 / 10000  # 手续费
    t_rate = 1 / 1000  # 印花税
    period = 'M'  # 选股周期
    date_start = '2010-01-01'  # 回测开始时间
    date_end = '2022-04-13'

    # ===导入数据
    df = pd.read_parquet('data/all_stock_data.parquet')
    df.dropna(subset=['next_every_day_return'],
              inplace=True)  # 删除最近一个周期，不删除可以做为实盘下周期选股结果

    # 导入指数数据
    # index_data = import_index_data(back_trader_start=date_start,
    #                                back_trader_end=date_end)
    index_data = pd.read_parquet('data/index.parquet')

    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, period)

    df = df[df['date'] >= pd.to_datetime(date_start)]
    # ===选股
    # 删除下个交易日不交易、开盘涨停的股票，因为这些股票在下个交易日开盘时不能买入。

    df = df[df['next_is_trade'] == 1]
    df = df[df['next_open_up'] == False]
    df = df[df['next_is_st'] == False]

    ###########################################################
    #用于选股，条件筛选
    # 计算选股因子，根据选股因子对股票进行排名
    df['rank'] = df.groupby('date')['circulating_market_cap'].rank()

    # 选取排名靠前的股票
    df = df[df['rank'] <= select_stock_num]
    ############################################################

    # 按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
    # 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
    df['next_open_return'] = df['next_open_return'].apply(lambda x: [x])
    df['next_every_day_return'] = df['next_every_day_return'].apply(
        lambda x: x[1:])

    df['next_every_day_return'] = df[[
        'next_open_return', 'next_every_day_return'
    ]].apply(lambda x: list(x[0]) + list(x[1]), axis=1)

    # ===整理选中股票数据
    # 挑选出选中股票
    df['symbol'] += ' '
    df['display_name'] += ' '
    group = df.groupby('date')
    select_stock = pd.DataFrame()
    select_stock['buy_stock_code'] = group['symbol'].sum()
    select_stock['buy_stock_name'] = group['display_name'].sum()

    # 计算下周期每天的资金曲线
    select_stock['next_every_day_equity_curve'] = group[
        'next_every_day_return'].apply(
            lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))

    # 扣除买入手续费
    select_stock['next_every_day_equity_curve'] = select_stock[
        'next_every_day_equity_curve'] * (1 - c_rate)  # 计算有不精准的地方
    # 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
    select_stock['next_every_day_equity_curve'] = select_stock[
        'next_every_day_equity_curve'].apply(
            lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

    # 计算下周期整体涨跌幅
    select_stock['next_return'] = select_stock[
        'next_every_day_equity_curve'].apply(lambda x: x[-1] - 1)

    # 计算下周期每天的涨跌幅
    select_stock['next_every_day_return'] = select_stock[
        'next_every_day_equity_curve'].apply(
            lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
    del select_stock['next_every_day_equity_curve']

    # 计算整体资金曲线
    select_stock.reset_index(inplace=True)
    select_stock['equity_curve'] = (select_stock['next_return'] + 1).cumprod()
    print(select_stock)
    select_stock.set_index('date', inplace=True)
    empty_df.update(select_stock)
    select_stock = empty_df
    select_stock.reset_index(inplace=True, drop=False)

    # ===计算选中股票每天的资金曲线
# 计算每日资金曲线

equity = pd.merge(left=index_data,
                  right=select_stock[['date', 'buy_stock_code']],
                  on=['date'],
                  how='left',
                  sort=True)

equity['hold_stock_code'] = equity['buy_stock_code'].shift()
equity['hold_stock_code'].fillna(method='ffill', inplace=True)
equity.dropna(subset=['hold_stock_code'], inplace=True)
del equity['buy_stock_code']

equity['return'] = select_stock['next_every_day_return'].sum()
equity['equity_curve'] = (equity['return'] + 1).cumprod()
equity['benchmark'] = (equity['index_return'] + 1).cumprod()
equity.to_parquet("data/result.parquet")