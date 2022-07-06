

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from numpy.fft import *
import math
from exchanges import BitfinexExch, CoinbaseProExch, BybitExch
from datetime import datetime
import requests
import hmac
import time

def ts_cross_val(model, x, y, cv=5, return_average=True):

    """
    Performs time series cross validation
    Parameters
    ----------
    model
    x: pd.Series, pd.DataFrame, np.array
        Predictive variables
    y: pd.Series, np.array
        target variable
    cv: int
        Number of splits
    return_average: boolean
        True - returns the average of the score for each split
        False - return a list with the scores for each split

    Returns
    ----------
    preds: pd.Series
        -- pandas series with predictions
    scores: list or np.average
        -- list with scores or averaged score
    """

    tscv = TimeSeriesSplit(n_splits=cv)
    preds = []
    test_indices = []
    scores = []
    indices = None

    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        indices = x.index.values
        x = np.array(x)

    if len(x.shape) == 1:
        x = np.array(x).reshape(-1, 1)

    for train_index, test_index in tscv.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        preds.append(model.predict(x_test))
        scores.append(model.score(x_test, y_test))
        if indices is not None:
            test_indices.append(indices[test_index])
        else:
            test_indices.append(test_index)
    preds = pd.Series(data=np.concatenate(preds), index=np.concatenate(test_indices))
    if return_average:
        return preds, np.average(scores)
    else:
        return preds, scores

def filter_signal(signal, threshold=1.5e5, dval = 20e-3, reflect = True, reflect_size = 30):

    if reflect:
        end = signal.iloc[-reflect_size:-1].sort_index(ascending = False)
        start = signal.iloc[1:reflect_size].sort_index(ascending = False)
        signal = pd.concat([start, signal, end], axis = 0)
        fourier = rfft(signal)
        frequencies = rfftfreq(signal.size, d=dval/signal.size)
        fourier[frequencies > threshold] = 0
        output = irfft(fourier)
        output = output[reflect_size-1:-reflect_size+1]
    else:
        fourier = rfft(signal)
        frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
        fourier[frequencies > threshold] = 0
        output = irfft(fourier)
    return output

def shift_dataset(series, lag = True, forecast = False, nlag = None, nforecast = None, var_lags = None, var_forecast = None, output_all = False, output_type = 'single_value', dropna = True, var_dict = None):
    """
    Shifts variables in dataset (forward, backward). Provides different methods for shifting variables forward, such as the mode output,
    which returns the mode of the next n periods.
    Parameters
    ----------
    lag: boolean
        True to shift variables backwards (lags).
    forecast: boolean
        True to shift variables forward (forecasts).
    nlag: int
        Lag period
    nforecast: int
        Forecast period
    var_lags: list
        If None, operations will be performed on the entire dataset. If a list is passed, only the variables in the list will be transformed
    var_forecast: list
        If None, operations will be performed on the entire dataset. If a list is passed,  only the variables in the list will be transformed
    output_all: boolean
        If True, range(forecasts) cols will be returned for each value in range, else, single output will be returned
    output_type: 'mode', 'mean', 'single_value'
        single_value -- X (t+nforecast) value will be returned
        mode -- the mode of the next nforecast values will be returned
        mean -- the mode of the next nforecast values will be returned
    dropna: boolean

    Returns
    ----------
        Pandas dataframe
    """

    output = pd.DataFrame(index = series.index)
    series_list = []
    columns = []
    if lag:
        if var_lags is None:
            for col in series.columns:
                for i in range(1, nlag):
                    series_list.append(series.loc[:, col].shift(i))
                    columns.append(col + ' t- ' + str(i))
        else:
            if var_dict is None:
                for var in var_lags:
                    for i in range(1, nlag):
                        series_list.append(series.loc[:, var].shift(i))
                        columns.append(var + ' t- ' + str(i))
            else:
                for key, value in var_dict.items():
                    for i in value:
                        series_list.append(series.loc[:, key].shift(i))
                        columns.append(key + ' t- ' + str(i))


    if forecast:
        if var_forecast is None:
            if output_all:
                if output_type == 'single_value':
                    for col in series.columns:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, col].shift(-i))
                            columns.append(col + ' t+ ' + str(i))
                elif output_type == 'mode':
                    for col in series.columns:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, col].rolling(window = i).apply(lambda x: stats.mode(x)[0]).shift(-i))
                            columns.append(col + ' t+ ' + str(i))
                elif output_type == 'mean':
                    for col in series.columns:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, col].rolling(window = i).apply(lambda x: np.mean(x)).shift(-i))
                            columns.append(col + ' t+ ' + str(i))
                elif output_type == 'std':
                    for col in series.columns:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, col].rolling(window = i).apply(lambda x: np.std(x)).shift(-i))
                            columns.append(col + ' t+ ' + str(i))
            else:
                if output_type == 'single_value':
                    for col in series.columns:
                        series_list.append(series.loc[:, col].shift(-nforecast))
                        columns.append(col + ' t+ ' + str(nforecast))
                elif output_type == 'mode':
                    for col in series.columns:
                        series_list.append(series.loc[:, col].rolling(window = nforecast).apply(lambda x: stats.mode(x)[0]).shift(-nforecast))
                        columns.append(col + ' t+ ' + str(nforecast))
                elif output_type == 'mean':
                    for col in series.columns:
                        series_list.append(series.loc[:, col].rolling(window = nforecast).apply(lambda x: np.mean(x)).shift(-nforecast))
                        columns.append(col + ' t+ ' + str(nforecast))
                elif output_type == 'std':
                    for col in series.columns:
                        series_list.append(series.loc[:, col].rolling(window = nforecast).apply(lambda x: np.std(x)).shift(-nforecast))
                        columns.append(col + ' t+ ' + str(nforecast))
        else:
            if output_all:
                if output_type == 'single_value':
                    for var in var_forecast:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, var].shift(-i))
                            columns.append(var + ' t+ ' + str(i))
                elif output_type == 'mode':
                    for var in var_forecast:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, var].rolling(window = i).apply(lambda x: stats.mode(x)[0]).shift(-i))
                            columns.append(var + ' t+ ' + str(i))
                elif output_type == 'mean':
                    for var in var_forecast:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, var].rolling(window = i).apply(lambda x: np.mean(x)).shift(-i))
                            columns.append(var + ' t+ ' + str(i))
                elif output_type == 'std':
                    for var in var_forecast:
                        for i in range(1, nforecast):
                            series_list.append(series.loc[:, var].rolling(window = i).apply(lambda x: np.std(x)).shift(-i))
                            columns.append(var + ' t+ ' + str(i))
            else:
                if output_type == 'single_value':
                    for var in var_forecast:
                        series_list.append(series.loc[:, var].shift(-nforecast))
                        columns.append(var + ' t+ ' + str(nforecast))
                elif output_type == 'mode':
                    for var in var_forecast:
                        series_list.append(series.loc[:, var].rolling(window = nforecast).apply(lambda x: stats.mode(x)[0]).shift(-nforecast))
                        columns.append(var + ' t+ ' + str(nforecast))
                elif output_type == 'mean':
                    for var in var_forecast:
                        series_list.append(series.loc[:, var].rolling(window = nforecast).apply(lambda x: np.mean(x)).shift(-nforecast))
                        columns.append(var + ' t+ ' + str(nforecast))
                elif output_type == 'std':
                    for var in var_forecast:
                        series_list.append(series.loc[:, var].rolling(window = nforecast).apply(lambda x: np.std(x)).shift(-nforecast))
                        columns.append(var + ' t+ ' + str(nforecast))

    series_list = pd.concat(series_list, axis=1)
    series_list.columns = columns
    output = pd.concat([series, series_list], axis = 1)

    if forecast:
        cols = output.columns.values.tolist()
        cols = cols[-1:] + cols[:-1]
        output = output[cols]
        output = output[[output.columns[0]] + sorted(output.columns[output.columns != output.columns[0]])]
    else:
        output = output[sorted(output.columns)]

    if dropna:
        return output.dropna()
    else:
        return output

def fetch_data(start, timeframe, symbol):

    gran = {'1h': 60*60, '1D': 60*60*24}
    csymbol = {'tBTCUSD': 'BTC-USD', 'tETHUSD': 'ETH-USD', 'tLTCUSD': 'LTCUSD'}

    btfx_exch = BitfinexExch('https://api-pub.bitfinex.com/v2/')
    cbse_exch = CoinbaseProExch('https://api.pro.coinbase.com/')
    bybit_exch = BybitExch('https://api.bybit.com')
    end = bybit_exch.server_time()

    ohlcv_data = btfx_exch.extract_ohlcv(timeframe=timeframe, symbol=symbol, section='hist', start=start, end=end)
    ohlcv_data.index = ohlcv_data.index.map(lambda x: datetime.utcfromtimestamp(int(str(x)[:-3])))
    ohlcv_data = ohlcv_data.loc['2016-01-01 00:00:00':, :]

    ohlcv_data_cbse = cbse_exch.extract_ohlcv(start=start, end=end, granularity=gran[timeframe], symbol=csymbol[symbol])
    ohlcv_data_cbse.index = ohlcv_data_cbse.index.map(lambda x: datetime.utcfromtimestamp(x))
    ohlcv_data_cbse = ohlcv_data_cbse.loc['2016-01-01 00:00:00':, :]

    ohlcv_data_cbse.columns = [x + '_cbs' for x in ohlcv_data_cbse.columns]
    ohlcv_data.columns = [x + '_bfx' for x in ohlcv_data.columns]
    complete_data = pd.concat([ohlcv_data, ohlcv_data_cbse], axis = 1)

    complete_data['Open_bfx'].fillna(complete_data['Open_cbs'], inplace=True)
    complete_data['Close_bfx'].fillna(complete_data['Close_cbs'], inplace=True)
    complete_data['High_bfx'].fillna(complete_data['High_cbs'], inplace=True)
    complete_data['Low_bfx'].fillna(complete_data['Low_cbs'], inplace=True)

    complete_data['Volume_bfx'] = complete_data['Volume_bfx'].fillna(0) + complete_data['Volume_cbs'].fillna(0)
    complete_data = complete_data.loc[:, list(filter(lambda x: 'bfx' in x, complete_data.columns))]
    complete_data.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    complete_data.dropna(inplace=True)

    return complete_data

# Exchange functions

def get_signature(private_key, req_params):
    _val = '&'.join([str(k)+"="+str(v) for k, v in sorted(req_params.items()) if (k != 'sign') and (v is not None)])
    return str(hmac.new(bytes(private_key, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256").hexdigest())

def server_time(base_endpoint):
    endpoint = '/v2/public/time'
    return(requests.get(url = base_endpoint + endpoint).json()['time_now'])

def wallet_balance(base_endpoint, api_key, private_key, coin):
    endpoint = '/v2/private/wallet/balance'
    params = {}
    params['api_key'] = api_key
    params['coin'] = coin
    params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
    params['sign'] = get_signature(private_key, params)
    return(requests.get(url = base_endpoint + endpoint, params = params))

def get_active_orders(symbol, base_endpoint, api_key, private_key):
    endpoint = '/v2/private/order/list'
    params = {}
    params['api_key'] = api_key
    params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
    params['symbol'] = symbol
    params['sign'] = get_signature(private_key, params)
    return(requests.get(url = base_endpoint + endpoint, params = params))

def get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_id):
    endpoint = '/v2/private/order'
    params = {}
    params['api_key'] = api_key
    params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
    params['symbol'] = symbol
    params['order_id'] = order_id
    params['sign'] = get_signature(private_key, params)
    return(requests.get(url = base_endpoint + endpoint, params = params))

def get_orderbook(base_endpoint, symbol):
    endpoint = '/v2/public/orderBook/L2'
    params = {}
    params['symbol'] = symbol
    return(requests.get(url = base_endpoint + endpoint, params = params))

def last_price(base_endpoint, symbol):
    endpoint = '/v2/public/tickers'
    params = {}
    params['symbol'] = symbol
    data = requests.get(url = base_endpoint + endpoint, params = params).json()
    last_price = float(data['result'][0]['last_price'])
    return(last_price)

def query_kline(base_endpoint, symbol, interval, from_timestamp):
    endpoint = '/v2/public/kline/list'
    params = {}
    params['symbol'] = symbol
    params['interval'] = interval
    params['from'] = from_timestamp
    data = requests.get(url = base_endpoint + endpoint, params = params).json()
    return data

def order_qty(base_endpoint, leverage, close_price, api_key, private_key):
    balance = {'BTC': 0, 'USDT': 0}
    for key, value in balance.items():
        balance[key] = value + wallet_balance(base_endpoint, api_key, private_key, coin = key).json()['result'][key]['equity']
    qty = 0
    qty = qty + (balance['BTC'] * close_price) * leverage
    qty = qty + (balance['USDT'] * leverage)
    return(int(qty))

def cancel_all_active_orders(base_endpoint, api_key, private_key, symbol):
    endpoint = '/v2/private/order/cancelAll'
    params = {}
    params['api_key'] = api_key
    params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
    params['symbol'] = symbol
    params['sign'] = get_signature(private_key, params)
    return(requests.post(url = base_endpoint + endpoint, data = params))

def get_positions(base_endpoint, api_key, private_key, symbol):
    endpoint = '/v2/private/position/list'
    params = {}
    params['api_key'] = api_key
    params['private_key'] = private_key
    params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
    params['symbol'] = symbol
    params['sign'] = get_signature(private_key, params)
    return(requests.get(url = base_endpoint + endpoint, params = params))

def place_order(base_endpoint, api_key, private_key, side, symbol, close_on_trigger, leverage=1, order_type='Limit', close_id=None):
    endpoint = '/v2/private/order/create'
    params = {}
    params['api_key'] = api_key
    params['side'] = side
    params['symbol'] = symbol
    params['order_type'] = order_type
    params['close_on_trigger'] = close_on_trigger
    params['recv_window'] = 10000
    if order_type == 'Limit':
        while True:
            params['time_in_force'] = 'PostOnly'
            if close_on_trigger:
                params['qty'] = get_active_orders(symbol, base_endpoint, api_key, private_key).json()['result']['data'][0]['qty']
                lop = None
            else:
                lp = round(last_price(base_endpoint, symbol), 0)
                params['qty'] = order_qty(base_endpoint, leverage, lp, api_key, private_key)
                close_order_status = 'Filled' if close_id == 'no_order' else get_active_orders_rt(symbol, base_endpoint, api_key, private_key, close_id).json()['result']
                if close_order_status == 'Filled':
                    is_filled = True
                else:
                    is_filled = close_order_status['order_status'] == 'Filled'
                if not is_filled:
                    lop = close_order_status['price']
                else:
                    lop = None
            if side == 'Buy':
                if lop is None:
                    lp = round(last_price(base_endpoint, symbol), 0)
                    open_at = lp - (lp * 0.07/100)
                else:
                    open_at = float(lop) - 0.5
            else:

                if lop is None:
                    lp = round(last_price(base_endpoint, symbol), 0)
                    open_at = lp + (lp * 0.07/100)
                else:
                    open_at = float(lop) + 0.5
            params['price'] = open_at
            params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
            params['sign'] = get_signature(private_key, params)
            rejected = requests.post(url = base_endpoint + endpoint, data = params).json()
            print(rejected)
            time.sleep(1)

            if rejected['ret_code'] == 30063:
                print('There is no open order to close')
                return 'no_order'
            if rejected['ret_code'] == 30007 or rejected['ret_code'] == 30005:
                print('Order price is out of permissible range, will open again')
                continue
            if rejected['result']['reject_reason'] != 'EC_NoError':
                print('Order did not enter orderbook')
                continue
            if rejected['result']['reject_reason'] == 'EC_NoError':
                print('Order was passed')
                return rejected['result']['order_id']
    else:
        params['time_in_force'] = 'GoodTillCancel'
        if not close_on_trigger:
            params['qty'] = order_qty(base_endpoint, leverage, last_price(base_endpoint, symbol), api_key, private_key)
        else:
            params['qty'] = get_active_orders(symbol, base_endpoint, api_key, private_key).json()['result']['data'][0]['qty']
        params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
        params['sign'] = get_signature(private_key, params)
        requests.post(url = base_endpoint + endpoint, data = params)

def place_active_order(base_endpoint, api_key, private_key, side, symbol, close_on_trigger, leverage = 1, order_type = 'Market',
                       order_to_close_id = None, size_to_close = None, closing_order_id = None):

    """
    Places order on bybit.
    Parameters:
    -- base_endpoint: bybit API base endpoint
    -- api_key: api key for the trading account
    -- private_key: private key for the trading account
    -- side: side of trade (either 'Buy' or 'Sell')
    -- symbol: cryptocurrency pair code ('BTCUSD')
    -- close_on_trigger: True to close open order, False to open new order
    -- leverage: position size multiplier (multiplies account's balance)
    -- order_type: either 'Limit' or 'Market'
    -- order_to_close_id: id representing the order that will be closed (if close_on_trigger = True)
    -- size_to_close:
    -- closing_order_id:
    """

    endpoint = '/v2/private/order/create'
    params = {}
    params['api_key'] = api_key
    params['side'] = side
    params['symbol'] = symbol
    params['order_type'] = order_type
    params['close_on_trigger'] = close_on_trigger
    params['recv_window'] = 8000
    if order_type == 'Market':
        params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
        params['time_in_force'] = 'GoodTillCancel'
        if close_on_trigger == False:
            params['qty'] = order_qty(base_endpoint, leverage, last_price(base_endpoint, symbol), api_key, private_key)
            order_link_id = str(int(float(server_time(base_endpoint)))) + 'o' + side
            params['order_link_id'] = order_link_id
        else:
            params['qty'] = get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_to_close_id).json()['result']['qty']
            order_link_id = str(int(float(server_time(base_endpoint)))) + 'c' + side
            params['order_link_id'] = order_link_id
        params['sign'] = get_signature(private_key, params)
        requests.post(url = base_endpoint + endpoint, data = params)
        return order_link_id
    else:
        params['time_in_force'] = 'PostOnly'
        if close_on_trigger == False:
            # To open new order
            params['qty'] = order_qty(base_endpoint, leverage, last_price(base_endpoint, symbol), api_key, private_key)
            print('Quantity: {}, close on trigger: {}, side: {}'.format(params['qty'], close_on_trigger, side))
        else:
            # To close open order
            if size_to_close is None:
                params['qty'] = get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_to_close_id).json()['result']['qty']
                print('Quantity: {}, close on trigger: {}, side: {}'.format(params['qty'], close_on_trigger, side))
            else:
                params['qty'] = size_to_close
        while True:
            last_price_order = round(last_price(base_endpoint, symbol), 0)
            if side == 'Buy' and close_on_trigger:
                order_link_id = str(int(float(server_time(base_endpoint)))) + 'cb'
                #order_link_id = order_to_close_id
                price = last_price_order - (last_price_order *0.1/100)
            if side == 'Sell' and close_on_trigger:
                order_link_id = str(int(float(server_time(base_endpoint)))) + 'cs'
                #order_link_id = order_to_close_id
                price = last_price_order + (last_price_order *0.1/100)
            if side == 'Buy' and not close_on_trigger:
                order_link_id = str(int(float(server_time(base_endpoint)))) + 'ob'
                active_order = get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_link_id).json()['result']
                price = last_price_order - (last_price_order *0.1/100)
                if active_order is not None:
                    if price == active_order['price']:
                        price = price - (last_price_order *0.03/100)
            if side == 'Sell' and not close_on_trigger:
                order_link_id = str(int(float(server_time(base_endpoint)))) + 'os'
                active_order = get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_link_id).json()['result']
                price = last_price_order + (last_price_order *0.1/100)
                if active_order is not None:
                    if price == active_order['price']:
                        price = price + (last_price_order *0.03/100)
            params['order_link_id'] = order_link_id
            params['price'] = price
            params['timestamp'] = int(server_time(base_endpoint).replace('.', '')[:13])
            params['sign'] = get_signature(private_key, params)
            rejected = requests.post(url = base_endpoint + endpoint, data = params).json()
            return rejected
            time.sleep(1)
            active_order = get_active_orders_rt(symbol, base_endpoint, api_key, private_key, order_link_id).json()['result']
            #print('API key: {}, Post request response: {}, active order response: {}'.format(api_key, rejected, active_order))
            if rejected['ret_code'] == 30007 or rejected['ret_code'] == 30005:
                print('Rejected code: {}'.format(rejected['ret_code']))
                print('')
                continue
            if active_order is None and close_on_trigger:
                print('There is no open order to close')
                print('')
                break
            if active_order is None and not close_on_trigger:
                print("Opened order is in orderbook and has not been filled")
                print('')
                return order_link_id
            if active_order['reject_reason'] != 'EC_NoError':
                print('Order did not enter orderbook')
                print('')
                continue
            if rejected['result']['reject_reason'] == 'EC_NoError':
                print('Order was passed')
                print('')
                return order_link_id
            else:
                print('Order was rejected')

def secs_till_next(endpoint):
    "Returns number of seconds until next hour"
    from datetime import timedelta, datetime
    delta = timedelta(hours=1)
    #now = datetime.fromtimestamp(get_atomic_time())
    now = datetime.fromtimestamp(float(server_time(endpoint)))
    next_minute = (now + delta).replace(microsecond=0, second=0, minute = 0)
    wait_seconds = (next_minute - now).seconds
    return(wait_seconds)
