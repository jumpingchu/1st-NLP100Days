# 匯入套件
from pandas_datareader import data as web
import yfinance as yf
import datetime as dt
import pandas as pd

stock_symbol = {
    "友達": 2409,
    "鴻海": 2317,
    "長榮": 2603, 
    "日月光投控": 3711,
    "晨訊科-DR": 912000}

# 取得yahoo股價
def get_stockdata(stock, t_start, t_end):
    yf.pdr_override()
    raw = 'stockdata.csv'

    ts = t_start.split("-")
    ts_y, ts_m, ts_d = int(ts[0]), int(ts[1]), int(ts[2])
    te = t_end.split("-")
    te_y, te_m, te_d = int(te[0]), int(te[1]), int(te[2])

    start = dt.datetime(ts_y, ts_m, ts_d)  # duration
    end = dt.datetime(te_y, te_m, te_d)
    df = web.get_data_yahoo([str(stock) + '.TW'], start, end)  # realtime
    df.to_csv(raw)

    return pd.read_csv(raw, parse_dates=True, index_col='Date')

# 取得收盤價
def get_stockinfo(stock_name, stock_data, price_category):
    price = stock_data[price_category]
    res = price.head()
    res = str(res).replace("Name: Close, dtype: float64",
                           "").replace("Date", "").strip("\n")
    return "日期" + " "*10 + "收盤價\n" + res


if __name__ == '__main__':

    stocks = [
        (2409, "友達"),
        (2317, "鴻海"),
        (2603, "長榮"),
        (3711, "日月光投控"),
        (912000, "晨訊科-DR")
    ]
    for stock_symbol, stock_name in stocks:
        stock_data = get_stockdata(stock_symbol, "2020-12-02", "2020-12-04")
        stock_info = get_stockinfo(stock_name, stock_data, "Close")
        print(stock_info)
        print("===")
